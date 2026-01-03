"""Training loop with dynamic token-aware accumulation."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from .data import DatasetArguments, load_and_prepare_dataset
from .train import build_model, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic token-aware fine-tuning")
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--mode", choices=("baseline", "lora", "qlora"), default="baseline")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["c_attn", "c_proj"])
    parser.add_argument("--target_tokens_per_step", type=int, help="Target number of tokens per optimizer step")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(no_cuda: bool) -> torch.device:
    if torch.cuda.is_available() and not no_cuda:
        return torch.device("cuda")
    if torch.backends.mps.is_available() and not no_cuda:
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloaders(args, tokenizer):
    data_args = DatasetArguments(
        dataset=args.dataset,
        block_size=args.block_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    datasets = load_and_prepare_dataset(data_args, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
    )
    eval_loader = DataLoader(
        datasets["validation"],
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
    )
    return train_loader, eval_loader


def checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir) / "dynamic_checkpoint.pt"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    batch_idx: int,
    global_step: int,
    token_tally: int,
):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "token_tally": token_tally,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    rng_state = state.get("rng_state")
    if rng_state is not None:
        if not isinstance(rng_state, torch.Tensor):
            rng_state = torch.tensor(rng_state, dtype=torch.uint8)
        rng_state = rng_state.detach().clone().to(torch.uint8).cpu()
        torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and "cuda_rng_state" in state:
        cuda_state = state["cuda_rng_state"]
        if isinstance(cuda_state, list):
            cuda_state = [
                cs.detach().clone().to(torch.uint8) if isinstance(cs, torch.Tensor) else torch.tensor(cs, dtype=torch.uint8)
                for cs in cuda_state
            ]
        else:
            cuda_state = cuda_state.detach().clone().to(torch.uint8)
        torch.cuda.set_rng_state_all(cuda_state)
    return state


def evaluate(model, dataloader, device) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def dynamic_train(args: argparse.Namespace):
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_model(args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    total, trainable, pct = count_trainable_parameters(model)
    print(f"Parameters total={total:,} trainable={trainable:,} ({pct:.2f}%)")

    train_loader, eval_loader = create_dataloaders(args, tokenizer)
    device = get_device(args.no_cuda)
    model.to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    steps_per_epoch = len(train_loader)
    max_steps = args.max_train_steps or int(args.num_train_epochs * steps_per_epoch)
    warmup_steps = int(max_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    ckpt_path = checkpoint_path(args.output_dir)
    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    token_tally = 0
    if args.resume_from_checkpoint and ckpt_path.exists():
        state = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        start_epoch = state.get("epoch", 0)
        start_batch_idx = state.get("batch_idx", 0)
        global_step = state.get("global_step", 0)
        token_tally = state.get("token_tally", 0)
        print(f"Resuming from checkpoint at epoch {start_epoch}, batch {start_batch_idx}, step {global_step}")

    target_tokens = args.target_tokens_per_step or (args.per_device_train_batch_size * args.block_size)
    print(f"Target tokens per optimizer step: {target_tokens}")

    running_loss = 0.0
    batches_since_log = 0
    start_time = time.perf_counter()
    total_epochs = math.ceil(args.num_train_epochs)
    eval_interval_steps = args.eval_interval

    for epoch in range(start_epoch, total_epochs):
        batch_skip = start_batch_idx if epoch == start_epoch else 0
        if batch_skip >= len(train_loader):
            continue
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < batch_skip:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            tokens = batch["input_ids"].numel()
            scaled_loss = loss * (tokens / target_tokens)
            scaled_loss.backward()
            token_tally += tokens
            running_loss += loss.item()
            batches_since_log += 1

            if token_tally >= target_tokens:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                token_tally = 0

                next_epoch = epoch
                next_batch_idx = batch_idx + 1
                if next_batch_idx >= len(train_loader):
                    next_epoch = epoch + 1
                    next_batch_idx = 0
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, next_epoch, next_batch_idx, global_step, token_tally
                )

                if global_step % args.log_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    avg_loss = running_loss / max(batches_since_log, 1)
                    toks_per_sec = target_tokens * global_step / max(elapsed, 1e-6)
                    print(
                        f"step {global_step} | loss {avg_loss:.4f} | tokens/sec {toks_per_sec:.1f}"
                    )
                    running_loss = 0.0
                    batches_since_log = 0

                if global_step % args.eval_interval == 0:
                    eval_loss = evaluate(model, eval_loader, device)
                    perplexity = math.exp(eval_loss)
                    print(
                        f"Eval step {global_step}: loss={eval_loss:.4f} perplexity={perplexity:.2f}"
                    )

                if global_step >= max_steps:
                    break
        start_batch_idx = 0
        if global_step >= max_steps:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    final_eval = evaluate(model, eval_loader, device)
    print(f"Training complete | final eval loss {final_eval:.4f} | perplexity {math.exp(final_eval):.2f}")


def main():
    args = parse_args()
    dynamic_train(args)


if __name__ == "__main__":
    main()
