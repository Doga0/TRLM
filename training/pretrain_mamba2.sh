#!/usr/bin/env bash
set -euo pipefail

# Hardware / token-budget settings. Keep these equal to the other baselines.
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-16}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
NUM_GPUS=${NUM_GPUS:-1}

MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1000000000}
EVAL_TOKEN_INTERVAL=${EVAL_TOKEN_INTERVAL:-10000000}
SAVE_TOKEN_INTERVAL=${SAVE_TOKEN_INTERVAL:-250000000}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-4000}

TOKENIZER_NAME=${TOKENIZER_NAME:-DogaOytc/TRLM-tokenizer}

# Mamba-2 target: approximately 75M unique parameters.
TARGET_PARAMS=${TARGET_PARAMS:-75000000}
HIDDEN_SIZE=${HIDDEN_SIZE:-512}
EXPAND=${EXPAND:-2}
STATE_SIZE=${STATE_SIZE:-128}
N_GROUPS=${N_GROUPS:-8}
HEAD_DIM=${HEAD_DIM:-64}
CONV_KERNEL=${CONV_KERNEL:-4}
CHUNK_SIZE=${CHUNK_SIZE:-256}
USE_BIAS=${USE_BIAS:-false}
USE_CONV_BIAS=${USE_CONV_BIAS:-true}
TIE_WORD_EMBEDDINGS=${TIE_WORD_EMBEDDINGS:-true}

VOCAB_SIZE=$(python - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${TOKENIZER_NAME}", trust_remote_code=False)
print(len(tok))
PY
)

# HF Mamba2 requires hidden_size * expand == num_heads * head_dim.
if [ -z "${NUM_HEADS:-}" ]; then
  INNER_SIZE=$((HIDDEN_SIZE * EXPAND))
  if [ $((INNER_SIZE % HEAD_DIM)) -ne 0 ]; then
    echo "ERROR: HIDDEN_SIZE*EXPAND (${INNER_SIZE}) must be divisible by HEAD_DIM (${HEAD_DIM})." >&2
    exit 1
  fi
  NUM_HEADS=$((INNER_SIZE / HEAD_DIM))
fi

# Select the layer count nearest to TARGET_PARAMS unless explicitly supplied.
if [ -z "${NUM_LAYERS:-}" ]; then
  NUM_LAYERS=$(python - <<PY
target = int("${TARGET_PARAMS}")
vocab = int("${VOCAB_SIZE}")
h = int("${HIDDEN_SIZE}")
expand = int("${EXPAND}")
state = int("${STATE_SIZE}")
groups = int("${N_GROUPS}")
head_dim = int("${HEAD_DIM}")
heads = int("${NUM_HEADS}")
conv_kernel = int("${CONV_KERNEL}")
use_bias = "${USE_BIAS}".lower() == "true"
use_conv_bias = "${USE_CONV_BIAS}".lower() == "true"
tied = "${TIE_WORD_EMBEDDINGS}".lower() == "true"

inner = h * expand
if inner != heads * head_dim:
    raise SystemExit(
        f"Invalid Mamba-2 dimensions: hidden*expand={inner}, heads*head_dim={heads*head_dim}"
    )
conv_dim = inner + 2 * groups * state
projection_size = inner + conv_dim + heads

# Exact unique parameters represented by the HF Mamba2 implementation.
per_layer = 0
per_layer += h                                  # block RMSNorm
per_layer += h * projection_size                # in_proj.weight
per_layer += projection_size if use_bias else 0 # in_proj.bias
per_layer += conv_dim * conv_kernel              # depthwise conv weight
per_layer += conv_dim if use_conv_bias else 0    # depthwise conv bias
per_layer += 3 * heads                           # dt_bias, A_log, D
per_layer += inner                               # gated RMSNorm
per_layer += inner * h                           # out_proj.weight
per_layer += h if use_bias else 0                # out_proj.bias

base = vocab * h                                 # token embedding
if not tied:
    base += vocab * h                            # independent LM head
base += h                                        # final RMSNorm

layers = max(1, int(round((target - base) / per_layer)))
print(layers)
PY
)
fi

APPROX_PARAMS=$(python - <<PY
vocab = int("${VOCAB_SIZE}")
h = int("${HIDDEN_SIZE}")
layers = int("${NUM_LAYERS}")
expand = int("${EXPAND}")
state = int("${STATE_SIZE}")
groups = int("${N_GROUPS}")
head_dim = int("${HEAD_DIM}")
heads = int("${NUM_HEADS}")
conv_kernel = int("${CONV_KERNEL}")
use_bias = "${USE_BIAS}".lower() == "true"
use_conv_bias = "${USE_CONV_BIAS}".lower() == "true"
tied = "${TIE_WORD_EMBEDDINGS}".lower() == "true"

inner = h * expand
if inner != heads * head_dim:
    raise SystemExit(
        f"Invalid Mamba-2 dimensions: hidden*expand={inner}, heads*head_dim={heads*head_dim}"
    )
conv_dim = inner + 2 * groups * state
projection_size = inner + conv_dim + heads
per_layer = (
    h
    + h * projection_size
    + (projection_size if use_bias else 0)
    + conv_dim * conv_kernel
    + (conv_dim if use_conv_bias else 0)
    + 3 * heads
    + inner
    + inner * h
    + (h if use_bias else 0)
)
base = vocab * h + h
if not tied:
    base += vocab * h
total = base + layers * per_layer
print(f"{total / 1_000_000:.2f}M")
PY
)

TOKENS_PER_STEP=$((PER_DEVICE_BATCH * GRAD_ACCUM * BLOCK_SIZE * NUM_GPUS))
MAX_STEPS=$((MAX_TRAIN_TOKENS / TOKENS_PER_STEP))
WARMUP_STEPS=$((MAX_STEPS * 3 / 100))
EVAL_STEPS=$((EVAL_TOKEN_INTERVAL / TOKENS_PER_STEP))
SAVE_STEPS=$((SAVE_TOKEN_INTERVAL / TOKENS_PER_STEP))

if [ "$MAX_STEPS" -lt 1 ]; then MAX_STEPS=1; fi
if [ "$EVAL_STEPS" -lt 1 ]; then EVAL_STEPS=1; fi
if [ "$SAVE_STEPS" -lt 1 ]; then SAVE_STEPS=1; fi
if [ "$WARMUP_STEPS" -lt 1 ]; then WARMUP_STEPS=1; fi

OUTPUT_DIR=${OUTPUT_DIR:-/content/drive/MyDrive/TRM/TRLM/exp4/mamba2/mamba2_checkpoints}
TENSORBOARD_LOGGING_DIR=${TENSORBOARD_LOGGING_DIR:-/content/drive/MyDrive/TRM/TRLM/exp4/mamba2/mamba2_logs}
export TENSORBOARD_LOGGING_DIR

echo "TOKENIZER_NAME=${TOKENIZER_NAME}"
echo "VOCAB_SIZE=${VOCAB_SIZE}"
echo "HIDDEN_SIZE=${HIDDEN_SIZE}"
echo "NUM_LAYERS=${NUM_LAYERS}"
echo "EXPAND=${EXPAND}"
echo "STATE_SIZE=${STATE_SIZE}"
echo "N_GROUPS=${N_GROUPS}"
echo "NUM_HEADS=${NUM_HEADS}"
echo "HEAD_DIM=${HEAD_DIM}"
echo "CONV_KERNEL=${CONV_KERNEL}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "TIE_WORD_EMBEDDINGS=${TIE_WORD_EMBEDDINGS}"
echo "APPROX_PARAMS=${APPROX_PARAMS}"
echo "TOKENS_PER_STEP=${TOKENS_PER_STEP}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "EVAL_STEPS=${EVAL_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"

python training/run_clm.py \
  --model_type mamba2 \
  --tokenizer_name "${TOKENIZER_NAME}" \
  --trust_remote_code false \
  --use_trlm_mix \
  --streaming \
  --openwebtext_dataset_name Skylion007/openwebtext \
  --finewebedu_dataset_name HuggingFaceFW/fineweb-edu \
  --finewebedu_dataset_config_name sample-10BT \
  --streaming_shuffle_buffer_size 10000 \
  --validation_split_percentage 1 \
  --block_size "${BLOCK_SIZE}" \
  --config_overrides "hidden_size=${HIDDEN_SIZE},state_size=${STATE_SIZE},num_hidden_layers=${NUM_LAYERS},num_heads=${NUM_HEADS},head_dim=${HEAD_DIM},expand=${EXPAND},conv_kernel=${CONV_KERNEL},n_groups=${N_GROUPS},use_bias=${USE_BIAS},use_conv_bias=${USE_CONV_BIAS},hidden_act=silu,layer_norm_epsilon=1e-5,initializer_range=0.1,residual_in_fp32=true,rescale_prenorm_residual=false,use_cache=false,rms_norm=true,chunk_size=${CHUNK_SIZE},tie_word_embeddings=${TIE_WORD_EMBEDDINGS}" \
  --do_train \
  --do_eval \
  --max_steps "${MAX_STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --logging_strategy steps \
  --logging_steps 50 \
  --eval_progress_update_steps 50 \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
  --per_device_eval_batch_size "${PER_DEVICE_BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate 3e-4 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-5 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --torch_compile false \
  --save_total_limit 5 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard
