#!/usr/bin/env bash
set -euo pipefail

# Adjust these three values for your hardware.
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-16}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
NUM_GPUS=${NUM_GPUS:-1}

# TRLM token-budget baseline.
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1000000000}
EVAL_TOKEN_INTERVAL=${EVAL_TOKEN_INTERVAL:-10000000}
SAVE_TOKEN_INTERVAL=${SAVE_TOKEN_INTERVAL:-250000000}

TOKENS_PER_STEP=$((PER_DEVICE_BATCH * GRAD_ACCUM * BLOCK_SIZE * NUM_GPUS))
MAX_STEPS=$((MAX_TRAIN_TOKENS / TOKENS_PER_STEP))
WARMUP_STEPS=$((MAX_STEPS * 3 / 100))
EVAL_STEPS=$((EVAL_TOKEN_INTERVAL / TOKENS_PER_STEP))
SAVE_STEPS=$((SAVE_TOKEN_INTERVAL / TOKENS_PER_STEP))

# Avoid invalid 0 values if you increase the batch/GPU count a lot.
if [ "$EVAL_STEPS" -lt 1 ]; then EVAL_STEPS=1; fi
if [ "$SAVE_STEPS" -lt 1 ]; then SAVE_STEPS=1; fi
if [ "$WARMUP_STEPS" -lt 1 ]; then WARMUP_STEPS=1; fi

echo "TOKENS_PER_STEP=${TOKENS_PER_STEP}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "EVAL_STEPS=${EVAL_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"

python training/run_clm.py \
  --model_type gpt2 \
  --tokenizer_name DogaOytc/TRLM-tokenizer \
  --trust_remote_code false \
  --use_trlm_mix \
  --streaming \
  --openwebtext_dataset_name Skylion007/openwebtext \
  --finewebedu_dataset_name HuggingFaceFW/fineweb-edu \
  --finewebedu_dataset_config_name sample-10BT \
  --streaming_shuffle_buffer_size 10000 \
  --validation_split_percentage 1 \
  --block_size ${BLOCK_SIZE} \
  --config_overrides "n_embd=512,n_layer=12,n_head=8,n_inner=2816,n_positions=1024,activation_function=gelu_new,resid_pdrop=0.0,embd_pdrop=0.0,attn_pdrop=0.0,layer_norm_epsilon=1e-6,use_cache=false" \
  --do_train \
  --do_eval \
  --max_steps ${MAX_STEPS} \
  --warmup_steps ${WARMUP_STEPS} \
  --eval_strategy steps \
  --eval_steps ${EVAL_STEPS} \
  --save_strategy steps \
  --save_steps ${SAVE_STEPS} \
  --logging_strategy steps \
  --logging_steps 1 \
  --max_eval_samples 4000 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH} \
  --per_device_eval_batch_size ${PER_DEVICE_BATCH} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --learning_rate 3e-4 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-5 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --torch_compile true \
  --save_total_limit 5 \
  --output_dir /content/drive/MyDrive/TRM/TRLM/exp4/gpt2/gpt2_checkpoints \
  --logging_dir /content/drive/MyDrive/TRM/TRLM/exp4/gpt2/gpt2_logs \
  --report_to tensorboard