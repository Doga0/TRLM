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

TOKENIZER_NAME=${TOKENIZER_NAME:-DogaOytc/TRLM-tokenizer}

# Gemma 4 text ~74-80M target.
# Default target 77M; tokenizer vocab'a göre intermediate_size otomatik seçilir.
TARGET_PARAMS=${TARGET_PARAMS:-75000000}
HIDDEN_SIZE=${HIDDEN_SIZE:-512}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-8}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}
HEAD_DIM=${HEAD_DIM:-64}

# Eğer kendin sabitlemek istersen örn:
# GEMMA_INTERMEDIATE_SIZE=2304 bash run_gemma4.sh
VOCAB_SIZE=$(python - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${TOKENIZER_NAME}", trust_remote_code=False)
print(len(tok))
PY
)

if [ -z "${GEMMA_INTERMEDIATE_SIZE:-}" ]; then
  GEMMA_INTERMEDIATE_SIZE=$(python - <<PY
target = int("${TARGET_PARAMS}")
vocab = int("${VOCAB_SIZE}")
h = int("${HIDDEN_SIZE}")
layers = int("${NUM_LAYERS}")
heads = int("${NUM_HEADS}")
kv = int("${NUM_KV_HEADS}")
head_dim = int("${HEAD_DIM}")

# Approx unique params with tied embeddings and PLE disabled.
# embedding: vocab*h
# attention/layer: q + k + v + o = h*(heads*head_dim + 2*kv*head_dim + heads*head_dim)
# MLP/layer: gate + up + down = 3*h*intermediate
# norms are small but included approximately.
attn_per_layer = h * (heads * head_dim + 2 * kv * head_dim + heads * head_dim)
norm_per_layer = 4 * h + 2 * head_dim + 1
base = vocab * h + layers * (attn_per_layer + norm_per_layer) + h

raw_i = (target - base) / (layers * 3 * h)
# GPU dostu olması için 128'in katına yuvarla.
i = max(512, int(round(raw_i / 128) * 128))
print(i)
PY
)
fi

APPROX_PARAMS=$(python - <<PY
vocab = int("${VOCAB_SIZE}")
h = int("${HIDDEN_SIZE}")
layers = int("${NUM_LAYERS}")
heads = int("${NUM_HEADS}")
kv = int("${NUM_KV_HEADS}")
head_dim = int("${HEAD_DIM}")
i = int("${GEMMA_INTERMEDIATE_SIZE}")

attn_per_layer = h * (heads * head_dim + 2 * kv * head_dim + heads * head_dim)
mlp_per_layer = 3 * h * i
norm_per_layer = 4 * h + 2 * head_dim + 1
total = vocab * h + layers * (attn_per_layer + mlp_per_layer + norm_per_layer) + h
print(f"{total/1_000_000:.2f}M")
PY
)

TOKENS_PER_STEP=$((PER_DEVICE_BATCH * GRAD_ACCUM * BLOCK_SIZE * NUM_GPUS))
MAX_STEPS=$((MAX_TRAIN_TOKENS / TOKENS_PER_STEP))
WARMUP_STEPS=$((MAX_STEPS * 3 / 100))
EVAL_STEPS=$((EVAL_TOKEN_INTERVAL / TOKENS_PER_STEP))
SAVE_STEPS=$((SAVE_TOKEN_INTERVAL / TOKENS_PER_STEP))

# Avoid invalid 0 values if you increase the batch/GPU count a lot.
if [ "$EVAL_STEPS" -lt 1 ]; then EVAL_STEPS=1; fi
if [ "$SAVE_STEPS" -lt 1 ]; then SAVE_STEPS=1; fi
if [ "$WARMUP_STEPS" -lt 1 ]; then WARMUP_STEPS=1; fi

echo "TOKENIZER_NAME=${TOKENIZER_NAME}"
echo "VOCAB_SIZE=${VOCAB_SIZE}"
echo "HIDDEN_SIZE=${HIDDEN_SIZE}"
echo "NUM_LAYERS=${NUM_LAYERS}"
echo "NUM_HEADS=${NUM_HEADS}"
echo "NUM_KV_HEADS=${NUM_KV_HEADS}"
echo "HEAD_DIM=${HEAD_DIM}"
echo "GEMMA_INTERMEDIATE_SIZE=${GEMMA_INTERMEDIATE_SIZE}"
echo "APPROX_PARAMS=${APPROX_PARAMS}"
echo "TOKENS_PER_STEP=${TOKENS_PER_STEP}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "EVAL_STEPS=${EVAL_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"

python training/run_clm.py \
  --model_type gemma4_text \
  --tokenizer_name ${TOKENIZER_NAME} \
  --trust_remote_code false \
  --use_trlm_mix \
  --streaming \
  --openwebtext_dataset_name Skylion007/openwebtext \
  --finewebedu_dataset_name HuggingFaceFW/fineweb-edu \
  --finewebedu_dataset_config_name sample-10BT \
  --streaming_shuffle_buffer_size 10000 \
  --validation_split_percentage 1 \
  --block_size ${BLOCK_SIZE} \
  --config_overrides "hidden_size=${HIDDEN_SIZE},intermediate_size=${GEMMA_INTERMEDIATE_SIZE},num_hidden_layers=${NUM_LAYERS},num_attention_heads=${NUM_HEADS},num_key_value_heads=${NUM_KV_HEADS},head_dim=${HEAD_DIM},global_head_dim=${HEAD_DIM},max_position_embeddings=${BLOCK_SIZE},hidden_activation=gelu_pytorch_tanh,rms_norm_eps=1e-6,attention_bias=false,attention_dropout=0.0,sliding_window=512,hidden_size_per_layer_input=0,use_cache=false,tie_word_embeddings=false" \
  --do_train \
  --do_eval \
  --max_steps ${MAX_STEPS} \
  --warmup_steps ${WARMUP_STEPS} \
  --eval_strategy steps \
  --eval_steps ${EVAL_STEPS} \
  --save_strategy steps \
  --save_steps ${SAVE_STEPS} \
  --logging_strategy steps \
  --logging_steps 50 \
  --eval_progress_update_steps 50 \
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
  --output_dir /content/drive/MyDrive/TRM/TRLM/exp4/gemma4/gemma4_text_checkpoints \
  --logging_dir /content/drive/MyDrive/TRM/TRLM/exp4/gemma4/gemma4_text_logs \
  --report_to tensorboard