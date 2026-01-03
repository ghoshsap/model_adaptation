# Model Adaptation & Customization

This workspace is a collection of reproducible scripts for adapting compact GPT-style language models using Hugging Face tooling:

1. **Baseline fine-tuning** of `distilgpt2` on either TinyStories or WikiText-2.
2. **LoRA (PEFT)** adaptation that reaches the same quality with far fewer trainable parameters.
3. **4-bit QLoRA** training via `bitsandbytes` to reduce memory while keeping accuracy.
4. **KV cache benchmarking** to demonstrate faster generation throughput when caching is enabled.
5. **Orthogonal butterfly adapters** (full-width and blockwise) so you can compare low-rank LoRA vs structured rotations.
6. **Simple serving endpoint** (FastAPI) that can load the baseline or adapter checkpoints. Optionally swap in vLLM if you have it installed.

> **Note**: The scripts are CPU-friendly for debugging but assume you can access a CUDA GPU for the LoRA/QLoRA experiments.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are working on macOS without an NVIDIA GPU you can still test the baseline pipeline by setting `--mode baseline --no_cuda`. QLoRA and memory benchmarks require CUDA + `bitsandbytes`.

## Fine-tuning entry point

`src/train.py` exposes a single CLI that covers all experiments.

Key arguments:

- `--dataset {tinystories,wikitext2}` – pick a dataset (TinyStories defaults). TinyStories is downloaded from `roneneldan/TinyStories`. WikiText-2 uses `wikitext`,`wikitext-2-raw-v1`.
- `--mode {baseline,lora,qlora,butterfly,butterfly_block}` – toggles how the model is prepared.
- `--output_dir PATH` – where checkpoints, logs, and adapter weights land.
- `--max_train_samples`/`--max_eval_samples` – optional caps for quick iterations.
- `--lora_r`, `--lora_alpha`, `--lora_dropout`, `--target_modules` – PEFT knobs.
- `--gradient_checkpointing`, `--fp16`, `--bf16` flags help control memory.

### Suggested command set

**Baseline (full fine-tune, ~82M trainable params):**
```bash
python -m src.train \
  --dataset tinystories \
  --mode baseline \
  --output_dir outputs/baseline \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --gradient_checkpointing
```

**LoRA (only ~1.5M trainable params, same quality):**
```bash
python -m src.train \
  --dataset tinystories \
  --mode lora \
  --output_dir outputs/lora \
  --per_device_train_batch_size 16 \
  --max_train_samples 10000 \
  --max_eval_samples 1000 \
  --learning_rate 1e-4 \
  --lora_r 16 --lora_alpha 32 \
  --target_modules c_attn c_proj \
  --gradient_checkpointing
```

**QLoRA (4-bit base + LoRA adapters, fits on a single 8GB GPU):**
```bash
python -m src.train \
  --dataset tinystories \
  --mode qlora \
  --output_dir outputs/qlora \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-4 \
  --gradient_checkpointing \
  --lora_r 16 --lora_alpha 32
```

**Butterfly adapters (full rotation layers):**
```bash
python -m src.train \
  --dataset tinystories \
  --mode butterfly \
  --output_dir outputs/butterfly \
  --max_train_samples 10000 \
  --max_eval_samples 1000 \
  --block_size 256 \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-4 \
  --butterfly_stages 6 \
  --target_modules c_attn c_proj \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --logging_steps 50
```

**Blockwise butterfly (three 256-dim subspaces):**
```bash
python -m src.train \
  --dataset tinystories \
  --mode butterfly_block \
  --output_dir outputs/butterfly_block256 \
  --max_train_samples 10000 \
  --max_eval_samples 1000 \
  --block_size 256 \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-4 \
  --butterfly_block_size 256 \
  --butterfly_stages 8 \
  --target_modules c_attn c_proj \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --logging_steps 50
```

Logs show parameter counts, peak GPU memory (if CUDA), and evaluation perplexity so you can compare quality vs resource use.

## Memory + throughput experiments

### KV cache speed benchmark

```bash
python -m src.benchmark_generation \
  --model_path outputs/baseline \
  --prompt "Tell me a short story about robots." \
  --max_new_tokens 128
```

The script runs twice (cache on/off) and prints tokens/sec for each.

### Paged attention demo

`src/paged_attention.py` implements a simple paged KV cache for manual generation so you can play with cache length vs throughput:

```bash
python -m src.paged_attention \
  --model_path distilgpt2 \
  --adapter_path outputs/lora_tuned \
  --prompt "Explain gravity to a kid:" \
  --max_new_tokens 80 \
  --page_size 64 \
  --max_pages 6 \
  --temperature 0.9 \
  --repetition_penalty 1.1
```

The script prints the completion plus tokens/sec.

### Memory footprint

`src/train.py` logs the number of trainable parameters plus an optional GPU memory snapshot before and after wrapping with LoRA/QLoRA. Capture these numbers to show how LoRA keeps quality with fewer params, while QLoRA slashes activation + optimizer memory by quantizing the base weights.

## Dynamic token-aware trainer

If you want to experiment with variable-length batches while keeping a roughly constant number of tokens per optimizer step, use `src/train_dynamic.py`. It implements a simple manual training loop that scales each micro-batch loss by its token count and only steps the optimizer once a configurable token budget is reached.

Example:

```bash
python -m src.train_dynamic \
  --dataset tinystories \
  --mode lora \
  --output_dir outputs/lora_dynamic \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --max_train_samples 10000 \
  --max_eval_samples 1000 \
  --block_size 256 \
  --target_tokens_per_step 512 \
  --learning_rate 1.5e-4 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 1 \
  --log_interval 50 \
  --eval_interval 200 \
  --resume_from_checkpoint
```

The script writes `dynamic_checkpoint.pt` after every optimizer step; pass `--resume_from_checkpoint` to continue from the latest checkpoint if a run is interrupted. It coexists with the original `src.train`, so you can choose whichever flow you prefer.

## Serving an endpoint

Start the FastAPI app after you finish training:

```bash
uvicorn src.serve:app --reload --port 8000 \
  --factory --env-file .env \
  -- --model_path outputs/qlora --adapter_path outputs/qlora
```

Send a request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Write a bedtime story about fireflies.",
        "max_new_tokens": 64,
        "temperature": 0.7,
        "use_cache": true
      }'
```

## Repository layout

```
slm/
├── README.md
├── requirements.txt
├── scripts/               # Ready-to-run CLI wrappers (LoRA, butterfly, etc.)
└── src/
    ├── __init__.py
    ├── benchmark_generation.py   # KV cache token/sec measurement
    ├── data.py                   # Dataset helpers + tokenization
    ├── serve.py                  # FastAPI generation endpoint
    └── train.py                  # Baseline/LoRA/QLoRA fine-tuning CLI
```
