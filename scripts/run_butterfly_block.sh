#!/bin/bash
set -euo pipefail
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
