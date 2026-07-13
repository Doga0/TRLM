#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain

import datasets
import evaluate
import torch
import torch.nn.functional as F
from datasets import IterableDataset, IterableDatasetDict, interleave_datasets, load_dataset
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_utils import has_length
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: str | None = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: str | None = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: str | None = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    eval_progress_update_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Evaluation progress bar kaç prediction step'te bir güncellensin. "
                "1 varsayılan HF davranışı, 50 daha seyrek çıktı verir."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: int | None = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    # TRLM baseline data mixture: 50% OpenWebText + 50% FineWeb-Edu.
    # This mode bypasses --dataset_name and directly builds a streaming mixed dataset.
    use_trlm_mix: bool = field(
        default=False, metadata={"help": "Use 50/50 OpenWebText + FineWeb-Edu streaming mixture."}
    )
    openwebtext_dataset_name: str = field(
        default="Skylion007/openwebtext", metadata={"help": "OpenWebText dataset name on the Hub."}
    )
    openwebtext_dataset_config_name: str | None = field(
        default=None, metadata={"help": "OpenWebText dataset config name, if any."}
    )
    finewebedu_dataset_name: str = field(
        default="HuggingFaceFW/fineweb-edu", metadata={"help": "FineWeb-Edu dataset name on the Hub."}
    )
    finewebedu_dataset_config_name: str | None = field(
        default="sample-10BT", metadata={"help": "FineWeb-Edu dataset config name."}
    )
    streaming_shuffle_buffer_size: int = field(
        default=10_000, metadata={"help": "Shuffle buffer size for each streaming source."}
    )
    mix_seed: int = field(
        default=42, metadata={"help": "Seed for shuffling and 50/50 interleaving."}
    )
    diffusion_min_noise: float = field(
        default=0.05,
        metadata={"help": "Minimum uniform-token corruption probability for DiffusionGemma training."},
    )
    diffusion_max_noise: float = field(
        default=1.0,
        metadata={"help": "Maximum uniform-token corruption probability for DiffusionGemma training."},
    )
    diffusion_self_conditioning_probability: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of using an additional no-grad first pass for DiffusionGemma "
                "self-conditioning. Start with 0.0 for stability; 0.5 costs roughly 50% more forwards."
            )
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if (
            not self.use_trlm_mix
            and self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError("Need either --use_trlm_mix, a dataset name, or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def split_streaming_dataset(
    full_streaming_dataset,
    validation_percentage: int = 5,
) -> IterableDatasetDict:
    """
    Splits a streaming dataset into
    training and validation IterableDatasets, and supports methods like .map(), .filter(),
    .take() and properties like .features on the resulting streams.

    Args:
        full_streaming_dataset (Dataset): The name of the dataset to load (e.g., "HuggingFaceFW/fineweb").
        validation_percentage (int): The proportion of the dataset to be used for validation split.

    Returns:
        IterableDatasetDict: An IterableDatasetDict containing two IterableDataset objects: (train_stream, validation_stream).
    """
    if not (0 < validation_percentage < 100):
        raise ValueError(
            f"validation_percentage must be between 0 and 100 (exclusive). Passed: {validation_percentage}"
        )

    def split_generator(is_train: bool):
        for i, example in enumerate(full_streaming_dataset):
            bucket = i % 100
            if is_train:
                # Example: validation_percentage=1 -> bucket 1..99 train, bucket 0 validation.
                if bucket >= validation_percentage:
                    yield example
            else:
                if bucket < validation_percentage:
                    yield example

    features = full_streaming_dataset.features
    train_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": True}, features=features)
    validation_stream = IterableDataset.from_generator(
        split_generator, gen_kwargs={"is_train": False}, features=features
    )

    return IterableDatasetDict({"train": train_stream, "validation": validation_stream})


def _load_streaming_text_dataset(
    dataset_name: str,
    dataset_config_name: str | None,
    *,
    cache_dir: str | None,
    token: str | None,
    trust_remote_code: bool,
    seed: int,
    shuffle_buffer_size: int,
):
    """Load a streaming dataset and normalize it to a single `text` column."""
    load_kwargs = {
        "split": "train",
        "cache_dir": cache_dir,
        "token": token,
        "streaming": True,
        "trust_remote_code": trust_remote_code,
    }
    if dataset_config_name is None:
        ds = load_dataset(dataset_name, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, dataset_config_name, **load_kwargs)

    if shuffle_buffer_size and shuffle_buffer_size > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

    features = ds.features
    remove_columns = [c for c in features if c != "text"] if features is not None else None

    def keep_text_only(example):
        return {"text": example.get("text", "")}

    if remove_columns:
        ds = ds.map(keep_text_only, remove_columns=remove_columns)
    else:
        ds = ds.map(keep_text_only)

    return ds


def build_trlm_mixed_streaming_dataset(data_args, model_args):
    """Build 50/50 OpenWebText + FineWeb-Edu streaming train/validation datasets."""
    if not data_args.streaming:
        raise ValueError("--use_trlm_mix requires --streaming true")

    owt = _load_streaming_text_dataset(
        data_args.openwebtext_dataset_name,
        data_args.openwebtext_dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        seed=data_args.mix_seed,
        shuffle_buffer_size=data_args.streaming_shuffle_buffer_size,
    )
    fwe = _load_streaming_text_dataset(
        data_args.finewebedu_dataset_name,
        data_args.finewebedu_dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        seed=data_args.mix_seed + 1,
        shuffle_buffer_size=data_args.streaming_shuffle_buffer_size,
    )

    mixed_train_stream = interleave_datasets(
        [owt, fwe],
        probabilities=[0.5, 0.5],
        seed=data_args.mix_seed,
    )

    return split_streaming_dataset(mixed_train_stream, data_args.validation_split_percentage)

def safe_exp(value):
    """Compute exp(value) safely for perplexity logging."""
    try:
        return math.exp(float(value))
    except (OverflowError, ValueError, TypeError):
        return float("inf")


def get_world_size(training_args: TrainingArguments) -> int:
    """Return distributed world size in a way that also works for single-GPU/CPU runs."""
    return max(1, int(getattr(training_args, "world_size", 1)))


def get_effective_train_batch_size(training_args: TrainingArguments) -> int:
    """Number of sequence chunks consumed per optimizer update."""
    return (
        int(training_args.per_device_train_batch_size)
        * int(training_args.gradient_accumulation_steps)
        * get_world_size(training_args)
    )


def add_perplexity(metrics: dict, loss_key: str, output_key: str) -> None:
    """Add perplexity=exp(loss) to a metrics dictionary if the loss exists."""
    if loss_key in metrics:
        metrics[output_key] = safe_exp(metrics[loss_key])


def _parse_config_override_value(value: str):
    """Parse a Hugging Face --config_overrides value into a Python value."""
    value = value.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _get_text_config_for_overrides(config):
    """Return the nested text config for multimodal DiffusionGemma-style configs."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    return config


def apply_config_overrides(config, overrides: str) -> None:
    """
    Apply --config_overrides safely to both flat text configs and nested configs.

    DiffusionGemmaConfig keeps language-model fields such as hidden_size,
    intermediate_size, num_hidden_layers, etc. under config.text_config.
    Some Gemma4Text-only keys may not exist in DiffusionGemmaTextConfig;
    harmless compatibility keys are skipped with a warning instead of crashing.
    """
    if overrides is None:
        return

    text_config = _get_text_config_for_overrides(config)

    # These are compatibility overrides used by Gemma4Text/Gemma causal-LM configs
    # in some Transformers versions, but they are not exposed by
    # DiffusionGemmaTextConfig. If they are absent, skipping them preserves the
    # model defaults and prevents scratch-config construction from crashing.
    ignored_if_missing = {
        "hidden_size_per_layer_input",
        "use_cache",
    }

    for item in overrides.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid config override item: {item!r}. Expected key=value.")

        key, value = item.split("=", 1)
        key = key.strip()
        parsed_value = _parse_config_override_value(value)

        # DiffusionGemmaTextConfig validates this field as Literal["all", "vision"] | None,
        # not as a boolean. Accept convenient true/false command-line values anyway.
        if key == "use_bidirectional_attention":
            if parsed_value is True:
                parsed_value = "all"
            elif parsed_value is False:
                parsed_value = None

        if hasattr(config, key):
            setattr(config, key, parsed_value)
        elif text_config is not config and hasattr(text_config, key):
            setattr(text_config, key, parsed_value)
        elif key in ignored_if_missing:
            logger.warning(
                "Skipping config override %s=%r because it is not present in %s/%s. "
                "This is expected for DiffusionGemmaTextConfig.",
                key,
                parsed_value,
                getattr(config, "model_type", None),
                getattr(text_config, "model_type", None),
            )
        else:
            raise ValueError(
                f"key {key!r} is not present in the top-level config or nested text_config. "
                f"Top-level model_type={getattr(config, 'model_type', None)!r}, "
                f"text model_type={getattr(text_config, 'model_type', None)!r}"
            )


def _set_if_present(obj, key: str, value) -> None:
    if obj is not None and hasattr(obj, key) and value is not None:
        setattr(obj, key, value)


def sync_tokenizer_ids_to_config(config, tokenizer, block_size: int | None = None) -> None:
    """Mirror tokenizer-dependent fields into both top-level and nested text configs."""
    targets = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        targets.append(text_config)

    for target in targets:
        _set_if_present(target, "vocab_size", len(tokenizer))
        # Token ids are also stored on the top-level DiffusionGemmaConfig for
        # Trainer/generation synchronization, even though the embedding lives
        # under text_config. PreTrainedConfig permits these attributes.
        if tokenizer.bos_token_id is not None:
            setattr(target, "bos_token_id", tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            setattr(target, "eos_token_id", tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            setattr(target, "pad_token_id", tokenizer.pad_token_id)
        if block_size is not None:
            _set_if_present(target, "max_position_embeddings", block_size)


def is_diffusion_gemma_config(config) -> bool:
    """Return True for the top-level DiffusionGemmaConfig."""
    return getattr(config, "model_type", None) == "diffusion_gemma"


def patch_diffusion_gemma_text_only_encoder() -> None:
    """
    Patch Transformers 5.13.x so DiffusionGemma can be instantiated with
    ``vision_config=None``.

    The released configuration class explicitly permits ``vision_config=None``,
    but DiffusionGemmaEncoderModel.__init__ still creates the vision tower and
    multimodal projector unconditionally. For text-only training we replace only
    that constructor; all original text forward/masking/cache methods remain in
    use, and no vision parameters are created or saved.
    """
    try:
        from transformers.models.diffusion_gemma import modeling_diffusion_gemma as dg_modeling
    except ImportError as exc:
        raise ImportError(
            "Could not import Transformers DiffusionGemma modeling code. "
            "Install a Transformers version that contains DiffusionGemma support."
        ) from exc

    encoder_cls = dg_modeling.DiffusionGemmaEncoderModel
    if getattr(encoder_cls, "_trlm_text_only_constructor_patch", False):
        return

    def text_only_encoder_init(self, config):
        dg_modeling.DiffusionGemmaPreTrainedModel.__init__(self, config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = dg_modeling.DiffusionGemmaEncoderTextModel(
            config=config.text_config
        )
        # Keep the public attributes explicit so accidental image input fails
        # clearly instead of silently creating an unused vision network.
        self.vision_tower = None
        self.embed_vision = None
        self.post_init()

    encoder_cls.__init__ = text_only_encoder_init
    encoder_cls._trlm_text_only_constructor_patch = True
    logger.info(
        "Applied text-only DiffusionGemma encoder patch: vision tower and "
        "multimodal projector will not be instantiated."
    )


def build_model_from_config(config, model_args):
    """Instantiate a model from config, including non-AutoModelForCausalLM configs.

    DiffusionGemmaConfig is not registered under AutoModelForCausalLM because
    it is a block-diffusion model, not a standard left-to-right causal LM.
    It must be instantiated with DiffusionGemmaForBlockDiffusion.
    """
    if is_diffusion_gemma_config(config):
        config.vision_config = None
        patch_diffusion_gemma_text_only_encoder()
        try:
            from transformers import DiffusionGemmaForBlockDiffusion
        except ImportError as exc:
            raise ImportError(
                "DiffusionGemmaForBlockDiffusion could not be imported. "
                "Install a Transformers version that includes DiffusionGemma support, "
                "for example the same version used by your logs: transformers>=5.11."
            ) from exc

        logger.info("Instantiating DiffusionGemmaForBlockDiffusion from scratch.")
        return DiffusionGemmaForBlockDiffusion(config)

    return AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)


def load_model_from_pretrained(model_args, config, dtype):
    """Load pretrained model, including DiffusionGemma checkpoints."""
    if is_diffusion_gemma_config(config):
        config.vision_config = None
        patch_diffusion_gemma_text_only_encoder()
        try:
            from transformers import DiffusionGemmaForBlockDiffusion
        except ImportError as exc:
            raise ImportError(
                "DiffusionGemmaForBlockDiffusion could not be imported. "
                "Install a Transformers version that includes DiffusionGemma support."
            ) from exc

        logger.info("Loading DiffusionGemmaForBlockDiffusion checkpoint: %s", model_args.model_name_or_path)
        return DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            dtype=dtype,
        )

    return AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        dtype=dtype,
    )

def _default_gemma4_layer_types(num_hidden_layers: int, sliding_window_pattern: int = 6) -> list[str]:
    """Build default 5:1 local/global attention pattern for a custom depth."""
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

    layer_types = [
        "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
        for i in range(num_hidden_layers)
    ]
    # Gemma architectures expect the last layer to be full attention.
    layer_types[-1] = "full_attention"
    return layer_types



def configure_diffusion_gemma_text_only(config) -> None:
    """Disable all vision-side configuration for text-only DiffusionGemma."""
    if not is_diffusion_gemma_config(config):
        return

    if getattr(config, "vision_config", None) is not None:
        logger.warning(
            "Discarding diffusion_gemma vision_config because this run is text-only."
        )
    config.vision_config = None


def normalize_gemma4_config(config) -> None:
    """
    Keep Gemma4Text and nested DiffusionGemma text configs internally consistent.
    """
    model_type = getattr(config, "model_type", None)
    if model_type not in ["gemma4_text", "diffusion_gemma"]:
        return

    target_config = _get_text_config_for_overrides(config)

    if model_type == "diffusion_gemma":
        configure_diffusion_gemma_text_only(config)

    if model_type == "diffusion_gemma":
        if hasattr(target_config, "num_experts") and getattr(target_config, "num_experts", None) is None:
            logger.warning(
                "diffusion_gemma text_config.num_experts is None; setting num_experts=1 "
                "for a dense single-expert baseline."
            )
            target_config.num_experts = 1

        if hasattr(target_config, "top_k_experts") and getattr(target_config, "top_k_experts", None) is None:
            target_config.top_k_experts = 1

        if hasattr(target_config, "moe_intermediate_size") and getattr(target_config, "moe_intermediate_size", None) is None:
            target_config.moe_intermediate_size = int(getattr(target_config, "intermediate_size"))

        if hasattr(target_config, "num_global_key_value_heads") and getattr(target_config, "num_global_key_value_heads", None) is None:
            # Keep global attention KV grouping consistent with normal GQA unless
            # the config or command line provides a separate value.
            target_config.num_global_key_value_heads = int(getattr(target_config, "num_key_value_heads"))

        # Keep ``use_bidirectional_attention=None`` for the encoder.
        # DiffusionGemma's prompt encoder is causal; its canvas decoder is
        # bidirectional unconditionally inside DiffusionGemmaDecoderTextAttention.
        # Setting this to "all" would incorrectly make the prompt encoder
        # bidirectional and would break block-autoregressive cache semantics.

    try:
        num_hidden_layers = int(getattr(target_config, "num_hidden_layers"))
    except (TypeError, ValueError, AttributeError):
        raise ValueError(
            f"{model_type} text config has invalid num_hidden_layers="
            f"{getattr(target_config, 'num_hidden_layers', None)!r}"
        )

    layer_types = getattr(target_config, "layer_types", None)
    valid_layer_types = {"sliding_attention", "full_attention"}
    should_rebuild_layer_types = False

    if layer_types is None:
        should_rebuild_layer_types = True
    else:
        layer_types = list(layer_types)
        if len(layer_types) != num_hidden_layers:
            logger.warning(
                "%s layer_types length (%s) does not match num_hidden_layers (%s); "
                "rebuilding layer_types with the default 5:1 sliding/full pattern.",
                model_type,
                len(layer_types),
                num_hidden_layers,
            )
            should_rebuild_layer_types = True
        elif any(layer_type not in valid_layer_types for layer_type in layer_types):
            logger.warning(
                "%s layer_types contains invalid values %s; rebuilding layer_types.",
                model_type,
                sorted(set(layer_types) - valid_layer_types),
            )
            should_rebuild_layer_types = True

    if should_rebuild_layer_types:
        target_config.layer_types = _default_gemma4_layer_types(num_hidden_layers)
    else:
        target_config.layer_types = layer_types
        if target_config.layer_types[-1] != "full_attention":
            logger.warning(
                "%s last layer must be full_attention, got %s; forcing it to full_attention.",
                model_type,
                target_config.layer_types[-1],
            )
            target_config.layer_types[-1] = "full_attention"

    if len(target_config.layer_types) != num_hidden_layers:
        raise ValueError(
            f"{model_type} config normalization failed: "
            f"num_hidden_layers={num_hidden_layers}, len(layer_types)={len(target_config.layer_types)}"
        )


class SparseEvalProgressCallback(ProgressCallback):
    """
    Default Hugging Face progress callback'inin eval progress bar güncellemesini seyrekleştirir.

    Normalde evaluation sırasında tqdm her prediction step'te update edilir.
    Bu callback ile örn. eval_progress_update_steps=50 yapınca progress bar 50 step'te bir güncellenir.
    """

    def __init__(self, eval_progress_update_steps: int = 1):
        super().__init__()
        self.eval_progress_update_steps = max(1, int(eval_progress_update_steps))
        self._pending_eval_updates = 0

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if self.eval_progress_update_steps <= 1:
            return super().on_prediction_step(
                args,
                state,
                control,
                eval_dataloader=eval_dataloader,
                **kwargs,
            )

        if state.is_local_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader),
                    leave=self.training_bar is None,
                    dynamic_ncols=True,
                )
                self._pending_eval_updates = 0

            self._pending_eval_updates += 1

            if self._pending_eval_updates >= self.eval_progress_update_steps:
                self.prediction_bar.update(self._pending_eval_updates)
                self._pending_eval_updates = 0

    def _flush_eval_updates(self):
        if self.prediction_bar is not None and self._pending_eval_updates > 0:
            self.prediction_bar.update(self._pending_eval_updates)
        self._pending_eval_updates = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self._flush_eval_updates()
        return super().on_evaluate(args, state, control, **kwargs)

    def on_predict(self, args, state, control, **kwargs):
        self._flush_eval_updates()
        return super().on_predict(args, state, control, **kwargs)

class TokenAccountingTrainer(Trainer):
    """
    Trainer with token accounting and a DiffusionGemma block-denoising objective.

    For DiffusionGemma, each fixed-length sample is split into:
      * causal encoder context: ``block_size - canvas_length`` tokens
      * clean decoder canvas: ``canvas_length`` tokens

    The clean canvas is corrupted with uniform random vocabulary tokens. The
    model predicts the clean canvas bidirectionally and cross-entropy is applied
    only at corrupted positions, with inverse corruption-probability weighting.
    """

    def __init__(
        self,
        *args,
        block_size: int,
        eval_samples: int | None = None,
        diffusion_min_noise: float = 0.05,
        diffusion_max_noise: float = 1.0,
        diffusion_self_conditioning_probability: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_size = int(block_size)
        self.effective_train_batch_size = get_effective_train_batch_size(self.args)
        self.tokens_per_train_step = self.effective_train_batch_size * self.block_size
        self.eval_samples = eval_samples
        self.eval_tokens = None if eval_samples is None else int(eval_samples) * self.block_size

        self.is_diffusion_gemma = is_diffusion_gemma_config(self.model.config)
        self.canvas_length = (
            int(getattr(self.model.config, "canvas_length", 256))
            if self.is_diffusion_gemma
            else 0
        )
        if self.is_diffusion_gemma:
            if not 0 < self.canvas_length < self.block_size:
                raise ValueError(
                    "DiffusionGemma requires 0 < canvas_length < block_size, got "
                    f"canvas_length={self.canvas_length}, block_size={self.block_size}."
                )
            text_config = getattr(self.model.config, "text_config", self.model.config)
            self.diffusion_vocab_size = int(text_config.vocab_size)
            self.diffusion_min_noise = float(diffusion_min_noise)
            self.diffusion_max_noise = float(diffusion_max_noise)
            self.diffusion_self_conditioning_probability = float(
                diffusion_self_conditioning_probability
            )
            if not 0.0 < self.diffusion_min_noise <= self.diffusion_max_noise <= 1.0:
                raise ValueError(
                    "Expected 0 < diffusion_min_noise <= diffusion_max_noise <= 1, got "
                    f"{self.diffusion_min_noise}, {self.diffusion_max_noise}."
                )
            if not 0.0 <= self.diffusion_self_conditioning_probability <= 1.0:
                raise ValueError(
                    "diffusion_self_conditioning_probability must be in [0, 1]."
                )
            # Trainer's prediction_step uses label_names to decide whether it
            # should call compute_loss during evaluation. Keep the dataset's
            # full clean sequence under `labels` and consume it here.
            self.label_names = ["labels"]
            self.target_tokens_per_sequence = self.canvas_length
        else:
            self.target_tokens_per_sequence = max(1, self.block_size - 1)

        self.loss_tokens_per_train_step = (
            self.effective_train_batch_size * self.target_tokens_per_sequence
        )
        self.eval_loss_tokens = (
            None
            if eval_samples is None
            else int(eval_samples) * self.target_tokens_per_sequence
        )
        self._eval_noise_generator = None

    def _train_input_tokens_seen(self) -> int:
        return int(self.state.global_step) * self.tokens_per_train_step

    def _train_loss_tokens_seen(self) -> int:
        return int(self.state.global_step) * self.loss_tokens_per_train_step

    def _rand(self, shape, *, device, dtype=torch.float32):
        generator = None if self.model.training else self._eval_noise_generator
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)

    def _randint(self, high: int, shape, *, device):
        generator = None if self.model.training else self._eval_noise_generator
        return torch.randint(0, high, shape, device=device, generator=generator)

    def _build_noisy_canvas(self, clean_canvas: torch.Tensor):
        batch_size, canvas_length = clean_canvas.shape
        device = clean_canvas.device

        if self.model.training:
            noise_probability = self._rand(
                (batch_size, 1), device=device, dtype=torch.float32
            )
            noise_probability = (
                self.diffusion_min_noise
                + noise_probability
                * (self.diffusion_max_noise - self.diffusion_min_noise)
            )
        else:
            # Fixed 50% corruption gives a stable validation target. The
            # generator is re-seeded at the start of every evaluate() call.
            noise_probability = torch.full(
                (batch_size, 1), 0.5, device=device, dtype=torch.float32
            )

        corruption_mask = self._rand(
            (batch_size, canvas_length), device=device, dtype=torch.float32
        ) < noise_probability

        # Guarantee at least one supervised position per sequence.
        empty_rows = ~corruption_mask.any(dim=1)
        if empty_rows.any():
            forced_positions = self._randint(
                canvas_length, (int(empty_rows.sum().item()),), device=device
            )
            corruption_mask[empty_rows, forced_positions] = True

        random_tokens = self._randint(
            self.diffusion_vocab_size, clean_canvas.shape, device=device
        )
        noisy_canvas = torch.where(corruption_mask, random_tokens, clean_canvas)
        return noisy_canvas, corruption_mask, noise_probability

    def _diffusion_compute_loss(self, model, inputs, return_outputs=False):
        # TrainingArguments in Transformers 5.13 has a use_cache field, but
        # DiffusionGemma explicitly rejects this argument because its decoder
        # always uses the encoder KV cache. Never forward it to the model.
        inputs = dict(inputs)
        inputs.pop("use_cache", None)
        inputs.pop("num_items_in_batch", None)

        clean_sequence = inputs.pop("labels", None)
        if clean_sequence is None:
            clean_sequence = inputs.get("input_ids")
        if clean_sequence is None:
            raise ValueError("DiffusionGemma training requires input_ids or labels.")

        clean_sequence = clean_sequence.long()
        if clean_sequence.shape[1] != self.block_size:
            raise ValueError(
                f"Expected sequence length {self.block_size}, got {clean_sequence.shape[1]}."
            )

        split_at = self.block_size - self.canvas_length
        encoder_input_ids = clean_sequence[:, :split_at]
        clean_canvas = clean_sequence[:, split_at:]

        full_attention_mask = inputs.get("attention_mask")
        encoder_attention_mask = (
            full_attention_mask[:, :split_at]
            if full_attention_mask is not None
            else None
        )

        noisy_canvas, corruption_mask, noise_probability = self._build_noisy_canvas(
            clean_canvas
        )

        model_kwargs = {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": noisy_canvas,
        }

        # Optional two-pass self-conditioning. The first-pass logits are detached
        # and fed to a second pass; no cache flag is supplied in either pass.
        use_self_conditioning = (
            model.training
            and self.diffusion_self_conditioning_probability > 0.0
            and torch.rand((), device=clean_sequence.device).item()
            < self.diffusion_self_conditioning_probability
        )
        if use_self_conditioning:
            with torch.no_grad():
                first_outputs = model(**model_kwargs)
            model_kwargs["self_conditioning_logits"] = first_outputs.logits.detach()
            model_kwargs["self_conditioning_mask"] = torch.ones(
                clean_sequence.shape[0],
                dtype=torch.bool,
                device=clean_sequence.device,
            )

        outputs = model(**model_kwargs)
        logits = outputs.logits
        if logits.shape[:2] != clean_canvas.shape:
            raise ValueError(
                "DiffusionGemma logits/canvas shape mismatch: "
                f"logits={tuple(logits.shape)}, target={tuple(clean_canvas.shape)}"
            )

        per_token_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            clean_canvas.reshape(-1),
            reduction="none",
        ).view_as(clean_canvas)

        # Inverse-probability weighting makes the random corruption estimator
        # comparable across sampled noise levels.
        weights = corruption_mask.float() / noise_probability.clamp_min(1e-4)
        loss = (per_token_loss * weights).sum() / weights.sum().clamp_min(1.0)

        if return_outputs:
            # Return only what Trainer needs; do not retain the large KV cache.
            return loss, {"loss": loss.detach(), "logits": logits}
        return loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        if self.is_diffusion_gemma:
            return self._diffusion_compute_loss(
                model, inputs, return_outputs=return_outputs
            )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )

    def evaluate(self, *args, **kwargs):
        if self.is_diffusion_gemma:
            device = next(self.model.parameters()).device
            self._eval_noise_generator = torch.Generator(device=device)
            self._eval_noise_generator.manual_seed(123456789)
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self._eval_noise_generator = None

    def add_token_accounting(self, logs: dict) -> dict:
        is_eval_log = any(key.startswith("eval_") for key in logs)

        logs["global_step"] = int(self.state.global_step)
        logs["train_input_tokens_seen"] = self._train_input_tokens_seen()
        logs["train_loss_tokens_seen"] = self._train_loss_tokens_seen()
        logs["train_tokens_seen_b"] = round(
            self._train_input_tokens_seen() / 1_000_000_000, 6
        )
        logs["tokens_per_train_step"] = self.tokens_per_train_step
        logs["loss_tokens_per_train_step"] = self.loss_tokens_per_train_step
        logs["effective_train_batch_size"] = self.effective_train_batch_size
        logs["sequence_length"] = self.block_size
        if self.is_diffusion_gemma:
            logs["diffusion_canvas_length"] = self.canvas_length
            logs["diffusion_context_length"] = self.block_size - self.canvas_length

        if is_eval_log and self.eval_samples is not None:
            logs["eval_samples_fixed"] = self.eval_samples
            logs["eval_input_tokens"] = self.eval_tokens
            logs["eval_loss_tokens"] = self.eval_loss_tokens

        if self.is_diffusion_gemma:
            add_perplexity(logs, "loss", "train_masked_perplexity")
            add_perplexity(logs, "eval_loss", "eval_masked_perplexity")
        else:
            add_perplexity(logs, "loss", "train_perplexity_from_loss")
            add_perplexity(logs, "eval_loss", "eval_perplexity")
        return logs

    def log(self, logs: dict, *args, **kwargs):
        logs = self.add_token_accounting(logs)
        return super().log(logs, *args, **kwargs)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.use_trlm_mix:
        logger.info("Using TRLM baseline data mixture: 50% OpenWebText + 50% FineWeb-Edu")
        raw_datasets = build_trlm_mixed_streaming_dataset(data_args, model_args)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    extension,
                    data_files=data_files,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )

                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            apply_config_overrides(config, model_args.config_overrides)
            logger.info(f"New config: {config}")

    normalize_gemma4_config(config)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # For scratch GPT-2 baseline, make the LM head/embedding size exactly match
    # the TRLM tokenizer. The original script only resized upward after model creation,
    # which can leave extra undecodable tokens when len(tokenizer) is smaller than
    # the default GPT-2 vocab size.
    if model_args.model_name_or_path is None:
        sync_tokenizer_ids_to_config(config, tokenizer, block_size=data_args.block_size)
        if data_args.block_size is not None and tokenizer.model_max_length < data_args.block_size:
            logger.warning(
                f"Tokenizer model_max_length={tokenizer.model_max_length} is smaller than "
                f"requested block_size={data_args.block_size}; overriding tokenizer.model_max_length "
                "for this scratch baseline."
            )
            tokenizer.model_max_length = data_args.block_size
        logger.info(
            "Scratch config vocab_size set to tokenizer length. "
            f"top_level={getattr(config, 'vocab_size', None)}, "
            f"text_config={getattr(getattr(config, 'text_config', None), 'vocab_size', None)}"
        )

    if model_args.model_name_or_path:
        dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        model = load_model_from_pretrained(model_args, config, dtype)
    else:
        model = build_model_from_config(config, model_args)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 1_000_000:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if is_diffusion_gemma_config(config):
        # DiffusionGemma always uses its encoder KV cache and rejects an explicit
        # use_cache argument. It is also currently incompatible with Inductor's
        # grouped-MoE fallback in Transformers/PyTorch, so keep this run eager.
        training_args.use_cache = None
        training_args.prediction_loss_only = True
        if training_args.torch_compile:
            logger.warning(
                "Disabling torch_compile for DiffusionGemma: grouped_mm MoE fallback "
                "causes repeated Inductor graph breaks in Transformers 5.13.1."
            )
            training_args.torch_compile = False
        if training_args.remove_unused_columns:
            logger.warning(
                "Setting remove_unused_columns=False so the clean labels remain "
                "available to the custom DiffusionGemma loss."
            )
            training_args.remove_unused_columns = False

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            if data_args.streaming:
                train_dataset = train_dataset.take(data_args.max_train_samples)
            else:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

    compute_metrics = None
    preprocess_logits_for_metrics = None
    eval_samples_for_accounting = None

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        eval_samples_for_accounting = None
        if data_args.max_eval_samples is not None:
            if data_args.streaming:
                # Materialize a fixed validation subset so every evaluation uses the same blocks
                # and so eval sample/token accounting is deterministic.
                logger.info(f"Materializing {data_args.max_eval_samples} validation blocks for fixed evaluation.")
                eval_dataset = datasets.Dataset.from_list(list(eval_dataset.take(data_args.max_eval_samples)))
                eval_samples_for_accounting = len(eval_dataset)
            else:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
                eval_samples_for_accounting = len(eval_dataset)
        elif not data_args.streaming:
            eval_samples_for_accounting = len(eval_dataset)

        if not is_diffusion_gemma_config(config):
            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)
        else:
            # Diffusion logits cover only the canvas, while labels contain the
            # full context+canvas sequence. The custom Trainer reports stable
            # masked denoising loss/perplexity directly.
            preprocess_logits_for_metrics = None
            compute_metrics = None

    # Initialize our Trainer
    trainer = TokenAccountingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
        block_size=block_size,
        eval_samples=eval_samples_for_accounting,
        diffusion_min_noise=data_args.diffusion_min_noise,
        diffusion_max_noise=data_args.diffusion_max_noise,
        diffusion_self_conditioning_probability=data_args.diffusion_self_conditioning_probability,
    )

    logger.info(
        "Token accounting: "
        f"effective_train_batch_size={trainer.effective_train_batch_size}, "
        f"block_size={block_size}, "
        f"tokens_per_train_step={trainer.tokens_per_train_step}, "
        f"loss_tokens_per_train_step={trainer.loss_tokens_per_train_step}, "
        f"planned_train_input_tokens={training_args.max_steps * trainer.tokens_per_train_step if training_args.max_steps > 0 else 'unknown'}"
    )

    if not training_args.disable_tqdm:
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(SparseEvalProgressCallback(data_args.eval_progress_update_steps))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        if data_args.streaming:
            # IterableDataset has no reliable len(); with streaming + max_steps, report
            # the number of fixed-length sequence chunks and tokens estimated from optimizer steps.
            metrics["train_samples"] = int(trainer.state.global_step) * trainer.effective_train_batch_size
        else:
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        metrics["train_input_tokens_seen"] = trainer._train_input_tokens_seen()
        metrics["train_loss_tokens_seen"] = trainer._train_loss_tokens_seen()
        metrics["train_tokens_seen_b"] = round(trainer._train_input_tokens_seen() / 1_000_000_000, 6)
        metrics["tokens_per_train_step"] = trainer.tokens_per_train_step
        metrics["loss_tokens_per_train_step"] = trainer.loss_tokens_per_train_step
        metrics["effective_train_batch_size"] = trainer.effective_train_batch_size
        metrics["sequence_length"] = block_size
        if trainer.is_diffusion_gemma:
            add_perplexity(metrics, "train_loss", "train_masked_perplexity")
        else:
            add_perplexity(metrics, "train_loss", "train_perplexity")

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        if eval_samples_for_accounting is not None:
            metrics["eval_samples"] = eval_samples_for_accounting
        elif data_args.streaming:
            metrics["eval_samples"] = data_args.max_eval_samples if data_args.max_eval_samples is not None else 0
        else:
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        metrics["eval_input_tokens"] = metrics["eval_samples"] * block_size
        metrics["eval_loss_tokens"] = (
            metrics["eval_samples"] * trainer.target_tokens_per_sequence
        )
        metrics["train_input_tokens_seen"] = trainer._train_input_tokens_seen()
        metrics["train_loss_tokens_seen"] = trainer._train_loss_tokens_seen()
        metrics["train_tokens_seen_b"] = round(trainer._train_input_tokens_seen() / 1_000_000_000, 6)
        if trainer.is_diffusion_gemma:
            add_perplexity(metrics, "eval_loss", "eval_masked_perplexity")
        else:
            add_perplexity(metrics, "eval_loss", "perplexity")
            metrics["eval_perplexity"] = metrics["perplexity"]

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()