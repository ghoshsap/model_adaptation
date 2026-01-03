#!/usr/bin/env bash
python -m src.train \
  --dataset tinystories \
  --mode butterfly \
  --output_dir outputs/butterfly_rotation \
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
