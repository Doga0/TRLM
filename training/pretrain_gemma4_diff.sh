#!/usr/bin/env bash
set -euo pipefail

PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-16}
BLOCK_SIZE=${BLOCK_SIZE:-1024}
CANVAS_LENGTH=${CANVAS_LENGTH:-256}
NUM_GPUS=${NUM_GPUS:-1}

MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-1000000000}
EVAL_TOKEN_INTERVAL=${EVAL_TOKEN_INTERVAL:-10000000}
SAVE_TOKEN_INTERVAL=${SAVE_TOKEN_INTERVAL:-250000000}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-4000}

TOKENIZER_NAME=${TOKENIZER_NAME:-DogaOytc/TRLM-tokenizer}
TARGET_PARAMS=${TARGET_PARAMS:-75000000}
HIDDEN_SIZE=${HIDDEN_SIZE:-512}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-8}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}
HEAD_DIM=${HEAD_DIM:-64}
NUM_EXPERTS=${NUM_EXPERTS:-1}
TOP_K_EXPERTS=${TOP_K_EXPERTS:-1}

# Start with one forward per step. Set to 0.5 later if you explicitly want
# stochastic two-pass self-conditioning (roughly 50% more forward compute).
DIFFUSION_SELF_CONDITIONING_PROBABILITY=${DIFFUSION_SELF_CONDITIONING_PROBABILITY:-0.0}
DIFFUSION_MIN_NOISE=${DIFFUSION_MIN_NOISE:-0.05}
DIFFUSION_MAX_NOISE=${DIFFUSION_MAX_NOISE:-1.0}

VOCAB_SIZE=$(python - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${TOKENIZER_NAME}", trust_remote_code=False)
print(len(tok))
PY
)

# Exact parameter model for the current text-only DiffusionGemma implementation:
# - tied token embedding / lm_head
# - one dense MLP per layer
# - one MoE expert MLP per layer (num_experts=1 does NOT disable MoE)
# - one decoder self-conditioning MLP
# - encoder/decoder common blocks share weights
if [ -z "${GEMMA_INTERMEDIATE_SIZE:-}" ]; then
  GEMMA_INTERMEDIATE_SIZE=$(python - <<PY
target = int("${TARGET_PARAMS}")
vocab = int("${VOCAB_SIZE}")
h = int("${HIDDEN_SIZE}")
layers = int("${NUM_LAYERS}")
heads = int("${NUM_HEADS}")
kv = int("${NUM_KV_HEADS}")
head_dim = int("${HEAD_DIM}")
num_experts = int("${NUM_EXPERTS}")

layer_types = [
    "sliding_attention" if (idx + 1) % 6 else "full_attention"
    for idx in range(layers)
]
layer_types[-1] = "full_attention"
num_sliding = layer_types.count("sliding_attention")
num_full = layer_types.count("full_attention")

q = h * (heads * head_dim)
k = h * (kv * head_dim)
v = h * (kv * head_dim)
o = (heads * head_dim) * h

# Full attention reuses K as V in DiffusionGemma; sliding attention has V projection.
sliding_attn = q + k + v + o + 2 * head_dim
full_attn = q + k + o + 2 * head_dim
attention_total = num_sliding * sliding_attn + num_full * full_attn

# Seven learned RMSNorm vectors + router projection/scale/per-expert scale.
static_per_layer = 7 * h + h * num_experts + h + num_experts
base = (
    vocab * h
    + attention_total
    + layers * static_per_layer
    + h  # final encoder norm
    + h  # self-conditioning pre-norm
)

# Dense MLP + expert MLPs in every shared layer, plus one self-conditioning MLP.
coefficient = (
    layers * 3 * h
    + layers * num_experts * 3 * h
    + 3 * h
)
raw_i = (target - base) / coefficient
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
num_experts = int("${NUM_EXPERTS}")

layer_types = [
    "sliding_attention" if (idx + 1) % 6 else "full_attention"
    for idx in range(layers)
]
layer_types[-1] = "full_attention"
num_sliding = layer_types.count("sliding_attention")
num_full = layer_types.count("full_attention")

q = h * (heads * head_dim)
k = h * (kv * head_dim)
v = h * (kv * head_dim)
o = (heads * head_dim) * h
sliding_attn = q + k + v + o + 2 * head_dim
full_attn = q + k + o + 2 * head_dim
attention_total = num_sliding * sliding_attn + num_full * full_attn

static_per_layer = 7 * h + h * num_experts + h + num_experts
mlp_total = layers * 3 * h * i
experts_total = layers * num_experts * 3 * h * i
self_conditioning = h + 3 * h * i

total = (
    vocab * h
    + attention_total
    + layers * static_per_layer
    + mlp_total
    + experts_total
    + h
    + self_conditioning
)
print(f"{total / 1_000_000:.2f}M")
PY
)

TOKENS_PER_STEP=$((PER_DEVICE_BATCH * GRAD_ACCUM * BLOCK_SIZE * NUM_GPUS))
MAX_STEPS=$((MAX_TRAIN_TOKENS / TOKENS_PER_STEP))
WARMUP_STEPS=$((MAX_STEPS * 3 / 100))
EVAL_STEPS=$((EVAL_TOKEN_INTERVAL / TOKENS_PER_STEP))
SAVE_STEPS=$((SAVE_TOKEN_INTERVAL / TOKENS_PER_STEP))

if [ "$EVAL_STEPS" -lt 1 ]; then EVAL_STEPS=1; fi
if [ "$SAVE_STEPS" -lt 1 ]; then SAVE_STEPS=1; fi
if [ "$WARMUP_STEPS" -lt 1 ]; then WARMUP_STEPS=1; fi

printf '%s\n' \
  "TOKENIZER_NAME=${TOKENIZER_NAME}" \
  "VOCAB_SIZE=${VOCAB_SIZE}" \
  "HIDDEN_SIZE=${HIDDEN_SIZE}" \
  "NUM_LAYERS=${NUM_LAYERS}" \
  "NUM_HEADS=${NUM_HEADS}" \
  "NUM_KV_HEADS=${NUM_KV_HEADS}" \
  "HEAD_DIM=${HEAD_DIM}" \
  "CANVAS_LENGTH=${CANVAS_LENGTH}" \
  "GEMMA_INTERMEDIATE_SIZE=${GEMMA_INTERMEDIATE_SIZE}" \
  "APPROX_PARAMS=${APPROX_PARAMS}" \
  "TOKENS_PER_STEP=${TOKENS_PER_STEP}" \
  "MAX_STEPS=${MAX_STEPS}" \
  "WARMUP_STEPS=${WARMUP_STEPS}" \
  "EVAL_STEPS=${EVAL_STEPS}" \
  "SAVE_STEPS=${SAVE_STEPS}"

# Transformers 5.13.1 DiffusionGemma rejects an explicit use_cache argument.
# torch.compile is disabled because the grouped-MoE fallback repeatedly breaks
# Inductor graphs (aten._local_scalar_dense) in the current implementation.
export TENSORBOARD_LOGGING_DIR=/content/drive/MyDrive/TRM/TRLM/exp4/gemma4/diffusion_gemma_logs

python training/run_clm.py \
  --model_type diffusion_gemma \
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
  --diffusion_min_noise "${DIFFUSION_MIN_NOISE}" \
  --diffusion_max_noise "${DIFFUSION_MAX_NOISE}" \
  --diffusion_self_conditioning_probability "${DIFFUSION_SELF_CONDITIONING_PROBABILITY}" \
  --config_overrides "hidden_size=${HIDDEN_SIZE},intermediate_size=${GEMMA_INTERMEDIATE_SIZE},moe_intermediate_size=${GEMMA_INTERMEDIATE_SIZE},num_experts=${NUM_EXPERTS},top_k_experts=${TOP_K_EXPERTS},num_hidden_layers=${NUM_LAYERS},num_attention_heads=${NUM_HEADS},num_key_value_heads=${NUM_KV_HEADS},num_global_key_value_heads=${NUM_KV_HEADS},head_dim=${HEAD_DIM},global_head_dim=${HEAD_DIM},max_position_embeddings=${BLOCK_SIZE},hidden_activation=gelu_pytorch_tanh,rms_norm_eps=1e-6,attention_bias=false,attention_dropout=0.0,sliding_window=512,use_bidirectional_attention=null,tie_word_embeddings=true,canvas_length=${CANVAS_LENGTH}" \
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
  --remove_unused_columns false \
  --prediction_loss_only true \
  --save_total_limit 5 \
  --output_dir /content/drive/MyDrive/TRM/TRLM/exp4/gemma4/diffusion_gemma_checkpoints \
  --report_to tensorboard
