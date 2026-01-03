"""Benchmarks generation throughput with and without a KV cache."""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV cache benchmark")
    parser.add_argument("--model_path", default="distilgpt2")
    parser.add_argument("--adapter_path")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(device)

    if args.adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to load adapters")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model.to(device)

    return model, tokenizer, device


def run_generation(model, tokenizer, device, prompt: str, use_cache: bool, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            temperature=temperature,
            top_p=top_p,
        )
        torch.cuda.synchronize() if device.startswith("cuda") else None
        elapsed = time.perf_counter() - start
    tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_second = tokens_generated / max(elapsed, 1e-6)
    return tokens_per_second, tokens_generated


def main():
    args = parse_args()
    model, tokenizer, device = load_model(args)

    for cache_state in (True, False):
        tps, count = run_generation(
            model,
            tokenizer,
            device,
            prompt=args.prompt,
            use_cache=cache_state,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        state = "enabled" if cache_state else "disabled"
        print(f"KV cache {state}: {tps:.2f} tokens/sec over {count} tokens")


if __name__ == "__main__":
    main()
