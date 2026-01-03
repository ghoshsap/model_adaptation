"""FastAPI endpoint for serving the fine-tuned models."""

from __future__ import annotations

import argparse
from functools import lru_cache
import time
import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.1, le=2.0)
    use_cache: bool = True


class ModelWrapper:
    def __init__(self, model_path: str, adapter_path: str | None, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        if adapter_path:
            if PeftModel is None:
                raise RuntimeError("peft is not installed; cannot load adapters")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def generate(self, req: GenerationRequest):
        inputs = self.tokenizer(req.prompt, return_tensors="pt").to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                use_cache=req.use_cache,
            )
        runtime = time.perf_counter() - start
        text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        completion = text[len(req.prompt) :].strip()
        generated_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
        tokens_per_second = generated_tokens / max(runtime, 1e-6)
        return completion, {
            "tokens_generated": generated_tokens,
            "runtime_seconds": runtime,
            "tokens_per_second": tokens_per_second,
        }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the fine-tuned model")
    parser.add_argument("--model_path", default="distilgpt2")
    parser.add_argument("--adapter_path")
    parser.add_argument("--device", default=None)
    return parser


def build_app(model_path: str, adapter_path: str | None, device: str | None) -> FastAPI:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = ModelWrapper(model_path, adapter_path, device)
    app = FastAPI(title="Small LM endpoint")

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device}

    @app.post("/generate")
    def generate(req: GenerationRequest):
        try:
            completion, stats = wrapper.generate(req)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"completion": completion, "stats": stats}

    return app


@lru_cache(maxsize=1)
def _cached_app():
    parser = create_parser()
    args, _ = parser.parse_known_args()
    model_path = args.model_path or os.getenv("SLM_MODEL_PATH", "distilgpt2")
    adapter_path = args.adapter_path or os.getenv("SLM_ADAPTER_PATH")
    device = args.device or os.getenv("SLM_DEVICE")
    return build_app(model_path, adapter_path, device)


def app():
    """Entry point used by ``uvicorn src.serve:app --factory``."""
    return _cached_app()
