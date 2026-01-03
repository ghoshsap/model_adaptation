#!/bin/bash
set -euo pipefail
python -m src.train \
  --dataset tinystories \
  --mode lora \
  --output_dir outputs/lora_run \
  --max_train_samples 10000 \
  --max_eval_samples 1000 \
  --block_size 256 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 1 \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --logging_steps 50
