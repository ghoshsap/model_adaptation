"""Fine-tune distilgpt2 with baseline, LoRA, or OSFT."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import inspect
from typing import Iterable, Tuple
import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.pytorch_utils import Conv1D

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .data import DatasetArguments, load_and_prepare_dataset
from .butterfly_adapter import ButterflyLinearAdapter, ButterflyConv1DAdapter
from .butterfly_rotation_adapter import apply_butterfly_rotation, ButterflyRotation
from .butterfly_block_adapter import (
    apply_blockwise_butterfly,
    BlockwiseButterflyRotation,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODES = ("baseline", "lora", "qlora", "butterfly", "butterfly_block")


def count_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total
    return total, trainable, pct


class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            logs["eval_perplexity"] = math.exp(logs["eval_loss"])
        return control


class JSONLoggingCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update(logs)
        with self.path.open("a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")
        return control


def wrap_with_butterfly(model, target_modules, stages):
    for name, module in model.named_children():
        matched = name in target_modules or any(name.endswith(t) for t in target_modules)
        if matched:
            if isinstance(module, torch.nn.Linear):
                setattr(model, name, ButterflyLinearAdapter(module, stages=stages))
                continue
            if isinstance(module, Conv1D):
                setattr(model, name, ButterflyConv1DAdapter(module, stages=stages))
                continue
        wrap_with_butterfly(module, target_modules, stages)


def get_device_map(no_cuda: bool) -> str | None:
    if no_cuda or not torch.cuda.is_available():
        return None
    return "auto"


def build_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_kwargs = {"device_map": get_device_map(args.no_cuda)}
    quant_config = None

    if args.mode == "qlora":
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs.setdefault("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if args.mode in {"lora", "qlora"}:
        if args.mode == "qlora":
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    elif args.mode == "butterfly":
        apply_butterfly_rotation(model, args.target_modules, args.butterfly_stages)
        for param in model.parameters():
            param.requires_grad = False
        for module in model.modules():
            if isinstance(module, ButterflyRotation):
                for param in module.parameters():
                    param.requires_grad = True
    elif args.mode == "butterfly_block":
        apply_blockwise_butterfly(
            model,
            args.target_modules,
            block_size=args.butterfly_block_size,
            stages=args.butterfly_stages,
        )
        for param in model.parameters():
            param.requires_grad = False
        for module in model.modules():
            if isinstance(module, BlockwiseButterflyRotation):
                for param in module.parameters():
                    param.requires_grad = True

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune distilgpt2 variants")
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--evaluation_strategy", default="steps")
    parser.add_argument("--save_strategy", default="steps")
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--merge_lora", action="store_true", help="merge adapters into base weights when saving")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["c_attn", "c_proj"],
    )
    parser.add_argument(
        "--butterfly_stages",
        type=int,
        default=None,
        help="Number of butterfly stages when mode=butterfly",
    )
    parser.add_argument(
        "--butterfly_block_size",
        type=int,
        default=256,
        help="Block size for blockwise butterfly mode",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id")
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    return parser


def log_gpu_memory(note: str):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info("%s | max CUDA memory %.2f GB", note, allocated)


def train(args: argparse.Namespace):
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model(args)
    total, trainable, pct = count_trainable_parameters(model)
    logger.info(
        "Parameters total=%s trainable=%s (%.2f%%)",
        f"{total:,}",
        f"{trainable:,}",
        pct,
    )

    data_args = DatasetArguments(
        dataset=args.dataset,
        block_size=args.block_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    datasets = load_and_prepare_dataset(data_args, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        report_to=None if args.report_to == "none" else args.report_to,
        evaluation_strategy=args.evaluation_strategy,
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    signature = inspect.signature(TrainingArguments.__init__)
    supports_eval_strategy = "evaluation_strategy" in signature.parameters
    supports_eval_strategy_short = "eval_strategy" in signature.parameters
    supports_save_strategy = "save_strategy" in signature.parameters
    evaluation_strategy = training_kwargs.get("evaluation_strategy")
    if not supports_eval_strategy and not supports_eval_strategy_short:
        logger.warning(
            "Installed transformers version does not accept `evaluation_strategy`; "
            "disabling load_best_model_at_end."
        )
        training_kwargs.pop("evaluation_strategy", None)
        training_kwargs.pop("eval_strategy", None)
        training_kwargs["load_best_model_at_end"] = False
    else:
        if not supports_eval_strategy:
            training_kwargs.pop("evaluation_strategy", None)
        if not supports_eval_strategy_short:
            training_kwargs.pop("eval_strategy", None)
    if not supports_save_strategy:
        logger.warning(
            "Installed transformers version does not accept `save_strategy`;"
            " falling back to default behavior."
        )
        training_kwargs.pop("save_strategy", None)
    training_args = TrainingArguments(**training_kwargs)
    if not supports_eval_strategy and hasattr(training_args, "evaluation_strategy"):
        setattr(training_args, "evaluation_strategy", evaluation_strategy)
    if not supports_eval_strategy_short and hasattr(training_args, "eval_strategy"):
        setattr(training_args, "eval_strategy", evaluation_strategy)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.add_callback(PerplexityCallback())
    trainer.add_callback(JSONLoggingCallback(Path(args.output_dir) / "log_history_full.json"))

    metrics = {}
    if not args.eval_only:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
    eval_metrics = trainer.evaluate()
    metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    perplexity = math.exp(eval_metrics["eval_loss"]) if "eval_loss" in eval_metrics else float("nan")
    metrics["perplexity"] = perplexity
    trainer.log_metrics("all", metrics)
    trainer.save_metrics("all", metrics)
    trainer.save_state()

    if args.merge_lora and args.mode in {"lora", "qlora"}:
        logger.info("Merging LoRA weights into the base model for export")
        model = trainer.model.merge_and_unload()
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    log_gpu_memory("Training complete")


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
