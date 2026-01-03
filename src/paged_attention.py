"""Paged attention + KV cache demo for autoregressive generation."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual generation loop with paged KV cache.")
    parser.add_argument("--model_path", default="distilgpt2")
    parser.add_argument("--adapter_path")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--page_size", type=int, default=128, help="Number of cached tokens per page.")
    parser.add_argument("--max_pages", type=int, default=4, help="Maximum number of pages to keep per layer.")
    parser.add_argument("--device")
    return parser


@dataclass
class GenerationStats:
    tokens_generated: int
    runtime: float

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_generated / max(self.runtime, 1e-6)


class PagedAttentionCache:
    """Maintains a sliding KV cache capped by page count."""

    def __init__(self, page_size: int, max_pages: int):
        self.page_size = page_size
        self.max_tokens = page_size * max_pages

    def trim(self, past_key_values):
        if past_key_values is None:
            return None
        trimmed = []
        for key, value in past_key_values:
            seq_len = key.shape[2]
            if seq_len > self.max_tokens:
                start = seq_len - self.max_tokens
                key = key[:, :, start:, :].contiguous()
                value = value[:, :, start:, :].contiguous()
            trimmed.append((key, value))
        return tuple(trimmed)


def prepare_model(model_path: str, adapter_path: str | None, device: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to load adapter checkpoints")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def logits_warper(top_k: int, top_p: float, temperature: float):
    warpers = LogitsProcessorList()
    if temperature != 1.0:
        warpers.append(lambda input_ids, scores: scores / temperature)
    if top_k > 0:
        warpers.append(TopKLogitsWarper(top_k))
    if 0 < top_p < 1:
        warpers.append(TopPLogitsWarper(top_p))
    return warpers


def apply_repetition_penalty(logits, generated, penalty: float):
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    for token_id in torch.unique(generated):
        logits[:, token_id] /= penalty
    return logits


def generate_with_paged_cache(model, tokenizer, device, args) -> tuple[str, GenerationStats]:
    cache = PagedAttentionCache(page_size=args.page_size, max_pages=args.max_pages)
    warpers = logits_warper(args.top_k, args.top_p, args.temperature)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    past_key_values = None
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            model_inputs = {
                "input_ids": generated[:, -1:] if past_key_values is not None else generated,
                "attention_mask": attention_mask,
                "use_cache": True,
                "past_key_values": past_key_values,
            }
            outputs = model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = apply_repetition_penalty(next_token_logits, generated, args.repetition_penalty)
            warped = next_token_logits
            for warper in warpers:
                warped = warper(generated, warped)
            probs = torch.softmax(warped, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = cache.trim(outputs.past_key_values)
    runtime = time.perf_counter() - start
    tokens_generated = generated.shape[-1] - inputs["input_ids"].shape[-1]
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    completion = text[len(args.prompt) :].strip()
    return completion, GenerationStats(tokens_generated=tokens_generated, runtime=runtime)


def main():
    parser = build_parser()
    args = parser.parse_args()
    model, tokenizer, device = prepare_model(args.model_path, args.adapter_path, args.device)
    completion, stats = generate_with_paged_cache(model, tokenizer, device, args)
    print(f"Paged attention completion ({stats.tokens_per_second:.2f} tok/s): {completion}")


if __name__ == "__main__":
    main()
