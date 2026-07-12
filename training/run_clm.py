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


def _default_gemma4_text_layer_types(num_hidden_layers: int, sliding_window_pattern: int = 6) -> list[str]:
    """Build Gemma4Text default 5:1 local/global attention pattern for a custom depth."""
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1 for gemma4_text, got {num_hidden_layers}")

    layer_types = [
        "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
        for i in range(num_hidden_layers)
    ]
    # Gemma4Text expects the last layer to be full attention.
    layer_types[-1] = "full_attention"
    return layer_types


def normalize_gemma4_text_config(config) -> None:
    """
    Keep Gemma4TextConfig internally consistent after --config_overrides.

    Gemma4TextConfig creates `layer_types` in __post_init__ using the default depth.
    If `num_hidden_layers` is later changed via config.update_from_string(), the old
    `layer_types` list can remain at the default length and save_pretrained() fails
    strict validation with:
        `num_hidden_layers` must be equal to the number of `layer_types`.

    This is intentionally scoped to gemma4_text so the same run_clm.py can still train
    other architectures without modifying their configs.
    """
    if getattr(config, "model_type", None) != "gemma4_text":
        return

    try:
        num_hidden_layers = int(getattr(config, "num_hidden_layers"))
    except (TypeError, ValueError):
        raise ValueError(
            f"gemma4_text config has invalid num_hidden_layers={getattr(config, 'num_hidden_layers', None)!r}"
        )

    layer_types = getattr(config, "layer_types", None)
    valid_layer_types = {"sliding_attention", "full_attention"}
    should_rebuild_layer_types = False

    if layer_types is None:
        should_rebuild_layer_types = True
    else:
        layer_types = list(layer_types)
        if len(layer_types) != num_hidden_layers:
            logger.warning(
                "gemma4_text layer_types length (%s) does not match num_hidden_layers (%s); "
                "rebuilding layer_types with the default 5:1 sliding/full pattern.",
                len(layer_types),
                num_hidden_layers,
            )
            should_rebuild_layer_types = True
        elif any(layer_type not in valid_layer_types for layer_type in layer_types):
            logger.warning(
                "gemma4_text layer_types contains invalid values %s; rebuilding layer_types.",
                sorted(set(layer_types) - valid_layer_types),
            )
            should_rebuild_layer_types = True

    if should_rebuild_layer_types:
        config.layer_types = _default_gemma4_text_layer_types(num_hidden_layers)
    else:
        config.layer_types = layer_types
        if config.layer_types[-1] != "full_attention":
            logger.warning(
                "gemma4_text last layer must be full_attention, got %s; forcing it to full_attention.",
                config.layer_types[-1],
            )
            config.layer_types[-1] = "full_attention"

    if len(config.layer_types) != num_hidden_layers:
        raise ValueError(
            "gemma4_text config normalization failed: "
            f"num_hidden_layers={num_hidden_layers}, len(layer_types)={len(config.layer_types)}"
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
    Trainer that adds token-budget metrics to every train/eval log.

    For causal LM, the model receives `block_size` input tokens per sequence,
    while the next-token loss is computed over approximately `block_size - 1`
    target tokens because labels are shifted internally by the model.
    """

    def __init__(self, *args, block_size: int, eval_samples: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = int(block_size)
        self.target_tokens_per_sequence = max(1, self.block_size - 1)
        self.effective_train_batch_size = get_effective_train_batch_size(self.args)
        self.tokens_per_train_step = self.effective_train_batch_size * self.block_size
        self.loss_tokens_per_train_step = self.effective_train_batch_size * self.target_tokens_per_sequence
        self.eval_samples = eval_samples
        self.eval_tokens = None if eval_samples is None else int(eval_samples) * self.block_size
        self.eval_loss_tokens = None if eval_samples is None else int(eval_samples) * self.target_tokens_per_sequence

    def _train_input_tokens_seen(self) -> int:
        return int(self.state.global_step) * self.tokens_per_train_step

    def _train_loss_tokens_seen(self) -> int:
        return int(self.state.global_step) * self.loss_tokens_per_train_step

    def add_token_accounting(self, logs: dict) -> dict:
        is_eval_log = any(key.startswith("eval_") for key in logs)

        logs["global_step"] = int(self.state.global_step)
        logs["train_input_tokens_seen"] = self._train_input_tokens_seen()
        logs["train_loss_tokens_seen"] = self._train_loss_tokens_seen()
        logs["train_tokens_seen_b"] = round(self._train_input_tokens_seen() / 1_000_000_000, 6)
        logs["tokens_per_train_step"] = self.tokens_per_train_step
        logs["loss_tokens_per_train_step"] = self.loss_tokens_per_train_step
        logs["effective_train_batch_size"] = self.effective_train_batch_size
        logs["sequence_length"] = self.block_size

        if is_eval_log and self.eval_samples is not None:
            logs["eval_samples_fixed"] = self.eval_samples
            logs["eval_input_tokens"] = self.eval_tokens
            logs["eval_loss_tokens"] = self.eval_loss_tokens

        # Training logs contain `loss`; eval logs contain `eval_loss`.
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

            manual_overrides = {}
            safe_overrides = []

            for item in model_args.config_overrides.split(","):
                key, value = item.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key == "n_inner":
                    manual_overrides[key] = int(value)
                else:
                    safe_overrides.append(f"{key}={value}")

            if safe_overrides:
                config.update_from_string(",".join(safe_overrides))

            for key, value in manual_overrides.items():
                setattr(config, key, value)

            logger.info(f"New config: {config}")

    normalize_gemma4_text_config(config)

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
        config.vocab_size = len(tokenizer)
        if tokenizer.bos_token_id is not None:
            config.bos_token_id = tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id
        if data_args.block_size is not None and tokenizer.model_max_length < data_args.block_size:
            logger.warning(
                f"Tokenizer model_max_length={tokenizer.model_max_length} is smaller than "
                f"requested block_size={data_args.block_size}; overriding tokenizer.model_max_length "
                "for this scratch GPT-2 baseline."
            )
            tokenizer.model_max_length = data_args.block_size
        logger.info(f"Scratch config vocab_size set to tokenizer length: {config.vocab_size}")

    if model_args.model_name_or_path:
        dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

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

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

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
        metrics["eval_loss_tokens"] = metrics["eval_samples"] * max(1, block_size - 1)
        metrics["train_input_tokens_seen"] = trainer._train_input_tokens_seen()
        metrics["train_loss_tokens_seen"] = trainer._train_loss_tokens_seen()
        metrics["train_tokens_seen_b"] = round(trainer._train_input_tokens_seen() / 1_000_000_000, 6)
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