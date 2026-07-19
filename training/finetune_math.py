#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune TRLM, GPT-2, Gemma4-Text, Mamba2, or DiffusionGemma on GSM8K + SVAMP and evaluate on their test splits.

TRLM mode:
    Prompt tokens stay visible.
    A fixed answer canvas is filled with [MASK] tokens.
    Loss is computed only on answer tokens.
    Evaluation uses TRLM mask-filling generation.
    By default, evaluation uses a fixed answer canvas instead of gold answer length.

GPT-2 / Gemma4-Text / Mamba2 mode:
    Prompt + answer tokens are fed autoregressively.
    Prompt labels are -100, answer labels are visible for causal LM training.
    Evaluation generates the answer from the prompt with greedy decoding.
    Mamba2 uses left padding, as recommended for batched Mamba2 execution.

DiffusionGemma mode:
    Prompt tokens are encoded causally as context.
    Answer tokens occupy the beginning of a fixed-size diffusion canvas.
    Random answer positions are corrupted and reconstructed bidirectionally.
    Evaluation uses DiffusionGemma's iterative block-denoising generate().

This lets you compare TRLM, GPT-2, Gemma4-Text, Mamba2, and DiffusionGemma on the same tokenizer,
same prompts, same target formatting, same splits, and same numeric exact-match metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)

try:
    from safetensors.torch import load_file as safetensors_load_file
except ImportError:  # pragma: no cover
    safetensors_load_file = None

# -----------------------------------------------------------------------------
# Make project imports work when the script is placed under training/ or scripts/
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name in {"scripts", "training"}:
    PROJECT_ROOT = SCRIPT_PATH.parents[1]
else:
    PROJECT_ROOT = SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import TRLMConfig  # noqa: E402
from models.trlm import TRLM  # noqa: E402


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: Any) -> Dict[str, Any]:
    try:
        return asdict(config)
    except TypeError:
        if hasattr(config, "to_dict"):
            return config.to_dict()
        if hasattr(config, "model_dump"):
            return config.model_dump()
        if hasattr(config, "dict"):
            return config.dict()
        return dict(config.__dict__)


def get_autocast_dtype(name: str) -> Optional[torch.dtype]:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def get_model_parameter_stats(model: torch.nn.Module, model_type: Optional[str] = None) -> Dict[str, Any]:
    """Return de-duplicated total/trainable parameter counts and memory footprint."""
    unique_parameters: Dict[int, torch.nn.Parameter] = {}
    for parameter in model.parameters():
        unique_parameters[id(parameter)] = parameter

    total_parameters = sum(parameter.numel() for parameter in unique_parameters.values())
    trainable_parameters = sum(
        parameter.numel() for parameter in unique_parameters.values() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in unique_parameters.values()
    )

    return {
        "model_type": model_type,
        "model_class": model.__class__.__name__,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "non_trainable_parameters": int(total_parameters - trainable_parameters),
        "total_parameters_millions": float(total_parameters / 1_000_000),
        "trainable_parameters_millions": float(trainable_parameters / 1_000_000),
        "parameter_memory_bytes": int(parameter_bytes),
        "parameter_memory_mib": float(parameter_bytes / (1024 ** 2)),
    }


def synchronize_device(device: torch.device) -> None:
    """Synchronize accelerator work so wall-clock generation timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def slice_batch_item(batch: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Create a batch of size one while preserving tensor and metadata fields."""
    single: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            single[key] = value[index : index + 1]
        elif isinstance(value, list):
            single[key] = value[index : index + 1]
        elif isinstance(value, tuple):
            single[key] = value[index : index + 1]
        else:
            single[key] = value
    return single


def normalize_number_text(x: Any) -> str:
    s = str(x).strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = s.strip().rstrip(".")
    return s


def extract_number(text: Any) -> Optional[float]:
    """Extract the final numeric answer from model text or dataset answer text."""
    if text is None:
        return None

    s = str(text)

    marker_patterns = [
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"Final\s*Answer\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"Answer\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    ]
    for pat in marker_patterns:
        matches = re.findall(pat, s, flags=re.IGNORECASE)
        if matches:
            try:
                return float(normalize_number_text(matches[-1]))
            except ValueError:
                pass

    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", s)
    if not nums:
        return None
    try:
        return float(normalize_number_text(nums[-1]))
    except ValueError:
        return None


def numeric_equal(pred: Optional[float], gold: Optional[float], tol: float = 1e-4) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


def is_causal_lm_type(model_type: str) -> bool:
    return model_type in {"gpt2", "gemma4_text", "mamba2"}


def is_mamba2_type(model_type: str) -> bool:
    return model_type == "mamba2"


def is_diffusion_lm_type(model_type: str) -> bool:
    return model_type == "diffusion_gemma"


def is_hf_pretrained_model_type(model_type: str) -> bool:
    return is_causal_lm_type(model_type) or is_diffusion_lm_type(model_type)


# -----------------------------------------------------------------------------
# Dataset formatting
# -----------------------------------------------------------------------------

def format_gsm8k(row: Dict[str, Any], target_mode: str) -> Dict[str, str]:
    question = str(row["question"]).strip()
    raw_answer = str(row["answer"]).strip()
    gold_num = extract_number(raw_answer)
    gold = str(int(gold_num)) if gold_num is not None and float(gold_num).is_integer() else str(gold_num)

    if "####" in raw_answer:
        reasoning, final = raw_answer.rsplit("####", 1)
        reasoning = reasoning.strip()
        gold = normalize_number_text(final)
    else:
        reasoning = raw_answer

    prompt = (
        "Solve the following grade-school math problem. "
        "End with 'Final Answer: <number>'.\n\n"
        f"Question: {question}\n\nAnswer:\n"
    )

    if target_mode == "final_only":
        target = f"Final Answer: {gold}\n"
    else:
        target = f"Solution:\n{reasoning}\nFinal Answer: {gold}\n"

    return {
        "source": "gsm8k",
        "prompt": prompt,
        "target": target,
        "gold_final": str(gold),
        "question": question,
    }


def _get_first_existing(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key])
    return default


def format_svamp(row: Dict[str, Any], target_mode: str) -> Dict[str, str]:
    body = _get_first_existing(row, ["Body", "body", "context", "Context"])
    question = _get_first_existing(row, ["Question", "question"])
    equation = _get_first_existing(row, ["Equation", "equation"], default="").strip()
    answer = _get_first_existing(row, ["Answer", "answer", "final_answer"], default="").strip()

    full_question = (body + " " + question).strip()
    gold_num = extract_number(answer)
    gold = normalize_number_text(answer)
    if gold_num is not None and float(gold_num).is_integer():
        gold = str(int(gold_num))

    prompt = (
        "Solve the following math word problem. "
        "Give the reasoning and end with 'Final Answer: <number>'.\n\n"
        f"Question: {full_question}\n\nAnswer:\n"
    )

    if target_mode == "final_only":
        target = f"Final Answer: {gold}\n"
    else:
        if equation:
            target = f"Equation: {equation}\nFinal Answer: {gold}\n"
        else:
            target = f"Final Answer: {gold}\n"

    return {
        "source": "svamp",
        "prompt": prompt,
        "target": target,
        "gold_final": str(gold),
        "question": full_question,
    }


class MathPromptDataset(Dataset):
    def __init__(
        self,
        hf_dataset: HFDataset,
        source: str,
        target_mode: str = "reasoning",
        max_examples: Optional[int] = None,
        split_name: str = "unknown",
    ) -> None:
        self.examples: List[Dict[str, Any]] = []
        self.source = str(source)
        self.split_name = str(split_name)
        n = len(hf_dataset) if max_examples is None else min(len(hf_dataset), max_examples)

        for dataset_index, row in enumerate(hf_dataset.select(range(n))):
            if source == "gsm8k":
                ex = format_gsm8k(row, target_mode=target_mode)
            elif source == "svamp":
                ex = format_svamp(row, target_mode=target_mode)
            else:
                raise ValueError(f"Unknown source: {source}")

            if extract_number(ex["gold_final"]) is not None:
                # Stable across model runs when the same dataset split and seed
                # are used. Prediction JSONL files can be joined by example_id.
                ex["dataset_split"] = self.split_name
                ex["dataset_index"] = int(dataset_index)
                ex["example_id"] = (
                    f"{self.source}_{self.split_name}_{dataset_index:05d}"
                )
                self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class MathCollator:
    def __init__(
        self,
        tokenizer,
        model_type: str,
        max_seq_len: int = 1024,
        max_answer_tokens: int = 384,
        pad_to_max_seq_len: bool = False,
        diffusion_canvas_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_seq_len = int(max_seq_len)
        self.max_answer_tokens = int(max_answer_tokens)
        self.pad_to_max_seq_len = bool(pad_to_max_seq_len)
        self.diffusion_canvas_length = int(diffusion_canvas_length)

        if model_type == "trlm" and tokenizer.mask_token_id is None:
            raise ValueError("TRLM mode requires tokenizer.mask_token_id.")
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")

        if is_diffusion_lm_type(model_type):
            if not 0 < self.diffusion_canvas_length < self.max_seq_len:
                raise ValueError(
                    "DiffusionGemma requires 0 < canvas_length < max_seq_len, got "
                    f"canvas_length={self.diffusion_canvas_length}, max_seq_len={self.max_seq_len}."
                )
            if self.max_answer_tokens > self.diffusion_canvas_length:
                print(
                    f"[WARN] max_answer_tokens={self.max_answer_tokens} exceeds DiffusionGemma "
                    f"canvas_length={self.diffusion_canvas_length}; answers will be capped to the canvas."
                )

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _encode_answer(self, text: str, hard_limit: int) -> List[int]:
        answer_ids = self._encode(text)
        if self.tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        answer_limit = min(self.max_answer_tokens, hard_limit)
        if len(answer_ids) > answer_limit:
            # Preserve the final answer marker and EOS when a reasoning trace is too long.
            answer_ids = answer_ids[-answer_limit:]
        return answer_ids

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, Any]:
        encoded_rows: List[Dict[str, Any]] = []

        for ex in features:
            prompt_ids = self._encode(ex["prompt"])

            if is_diffusion_lm_type(self.model_type):
                canvas_length = self.diffusion_canvas_length
                answer_ids = self._encode_answer(ex["target"], canvas_length)
                max_context_len = self.max_seq_len - canvas_length
                if max_context_len < 8:
                    raise ValueError(
                        f"max_seq_len={self.max_seq_len} leaves fewer than 8 prompt tokens "
                        f"after reserving canvas_length={canvas_length}."
                    )
                if len(prompt_ids) > max_context_len:
                    prompt_ids = prompt_ids[-max_context_len:]

                answer_len = len(answer_ids)
                clean_canvas = answer_ids + [self.tokenizer.pad_token_id] * (canvas_length - answer_len)
                canvas_labels = answer_ids + [-100] * (canvas_length - answer_len)

                encoded_rows.append({
                    "input_ids": prompt_ids,
                    "labels": canvas_labels,
                    "clean_canvas": clean_canvas,
                    "attention_mask": [1] * len(prompt_ids),
                    "prompt_len": len(prompt_ids),
                    "answer_len": answer_len,
                    "source": ex["source"],
                    "dataset_split": ex["dataset_split"],
                    "dataset_index": ex["dataset_index"],
                    "example_id": ex["example_id"],
                    "gold_final": ex["gold_final"],
                    "question": ex["question"],
                    "target_text": ex["target"],
                })
                continue

            min_prompt_tokens = 8

            if self.model_type == "trlm":
                answer_canvas_len = min(
                    self.max_answer_tokens,
                    self.max_seq_len - min_prompt_tokens,
                )
                answer_ids = self._encode_answer(ex["target"], answer_canvas_len)
                available_prompt_len = self.max_seq_len - answer_canvas_len
            else:
                answer_ids = self._encode_answer(ex["target"], self.max_seq_len)
                available_prompt_len = self.max_seq_len - len(answer_ids)
                if available_prompt_len < min_prompt_tokens:
                    answer_ids = answer_ids[-(self.max_seq_len - min_prompt_tokens):]
                    available_prompt_len = self.max_seq_len - len(answer_ids)

            if len(prompt_ids) > available_prompt_len:
                prompt_ids = prompt_ids[-available_prompt_len:]

            if self.model_type == "trlm":
                input_ids = prompt_ids + [self.tokenizer.mask_token_id] * answer_canvas_len
                labels = (
                    [-100] * len(prompt_ids)
                    + answer_ids
                    + [-100] * (answer_canvas_len - len(answer_ids))
                )
            elif is_causal_lm_type(self.model_type):
                input_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + answer_ids
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

            attention_mask = [1] * len(input_ids)

            encoded_rows.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "prompt_len": len(prompt_ids),
                "answer_len": len(answer_ids),
                "source": ex["source"],
                "dataset_split": ex["dataset_split"],
                "dataset_index": ex["dataset_index"],
                "example_id": ex["example_id"],
                "gold_final": ex["gold_final"],
                "question": ex["question"],
                "target_text": ex["target"],
            })

        if is_diffusion_lm_type(self.model_type):
            if self.pad_to_max_seq_len:
                context_batch_len = self.max_seq_len - self.diffusion_canvas_length
            else:
                context_batch_len = max(len(row["input_ids"]) for row in encoded_rows)

            input_batch, mask_batch = [], []
            for row in encoded_rows:
                # Left padding keeps the final real prompt token adjacent to the canvas.
                pad_len = context_batch_len - len(row["input_ids"])
                if pad_len < 0:
                    row["input_ids"] = row["input_ids"][-context_batch_len:]
                    row["attention_mask"] = row["attention_mask"][-context_batch_len:]
                    row["prompt_len"] = context_batch_len
                    pad_len = 0

                input_batch.append(
                    [self.tokenizer.pad_token_id] * pad_len + row["input_ids"]
                )
                mask_batch.append([0] * pad_len + row["attention_mask"])

            input_tensor = torch.tensor(input_batch, dtype=torch.long)
            return {
                "input_ids": input_tensor,
                "masked_input_ids": input_tensor,
                "labels": torch.tensor([r["labels"] for r in encoded_rows], dtype=torch.long),
                "clean_canvas": torch.tensor([r["clean_canvas"] for r in encoded_rows], dtype=torch.long),
                "attention_mask": torch.tensor(mask_batch, dtype=torch.bool),
                "prompt_lens": torch.tensor([r["prompt_len"] for r in encoded_rows], dtype=torch.long),
                "answer_lens": torch.tensor([r["answer_len"] for r in encoded_rows], dtype=torch.long),
                "sources": [r["source"] for r in encoded_rows],
                "dataset_splits": [r["dataset_split"] for r in encoded_rows],
                "dataset_indices": [r["dataset_index"] for r in encoded_rows],
                "example_ids": [r["example_id"] for r in encoded_rows],
                "gold_finals": [r["gold_final"] for r in encoded_rows],
                "questions": [r["question"] for r in encoded_rows],
                "target_texts": [r["target_text"] for r in encoded_rows],
            }

        if self.pad_to_max_seq_len:
            batch_len = self.max_seq_len
        else:
            batch_len = min(self.max_seq_len, max(len(r["input_ids"]) for r in encoded_rows))

        input_batch, label_batch, mask_batch = [], [], []
        for row in encoded_rows:
            pad_len = batch_len - len(row["input_ids"])
            if pad_len < 0:
                # Mamba2 should preserve the answer-side tail when truncation is
                # ever necessary. Other causal baselines retain the old behavior.
                if is_mamba2_type(self.model_type):
                    row["input_ids"] = row["input_ids"][-batch_len:]
                    row["labels"] = row["labels"][-batch_len:]
                    row["attention_mask"] = row["attention_mask"][-batch_len:]
                else:
                    row["input_ids"] = row["input_ids"][:batch_len]
                    row["labels"] = row["labels"][:batch_len]
                    row["attention_mask"] = row["attention_mask"][:batch_len]
                pad_len = 0

            if is_mamba2_type(self.model_type):
                # Hugging Face's Mamba2 implementation is primarily tested with
                # left padding. Right padding can propagate recurrent-state noise.
                input_batch.append([self.tokenizer.pad_token_id] * pad_len + row["input_ids"])
                label_batch.append([-100] * pad_len + row["labels"])
                mask_batch.append([0] * pad_len + row["attention_mask"])
            else:
                input_batch.append(row["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
                label_batch.append(row["labels"] + [-100] * pad_len)
                mask_batch.append(row["attention_mask"] + [0] * pad_len)

        input_tensor = torch.tensor(input_batch, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "masked_input_ids": input_tensor,
            "labels": torch.tensor(label_batch, dtype=torch.long),
            "attention_mask": torch.tensor(mask_batch, dtype=torch.long),
            "prompt_lens": torch.tensor([r["prompt_len"] for r in encoded_rows], dtype=torch.long),
            "answer_lens": torch.tensor([r["answer_len"] for r in encoded_rows], dtype=torch.long),
            "sources": [r["source"] for r in encoded_rows],
            "dataset_splits": [r["dataset_split"] for r in encoded_rows],
            "dataset_indices": [r["dataset_index"] for r in encoded_rows],
            "example_ids": [r["example_id"] for r in encoded_rows],
            "gold_finals": [r["gold_final"] for r in encoded_rows],
            "questions": [r["question"] for r in encoded_rows],
            "target_texts": [r["target_text"] for r in encoded_rows],
        }


# -----------------------------------------------------------------------------
# Split loading
# -----------------------------------------------------------------------------

def split_train_val(ds: HFDataset, val_ratio: float, seed: int) -> Tuple[HFDataset, HFDataset]:
    if val_ratio <= 0:
        return ds, ds.select(range(min(128, len(ds))))
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def load_gsm8k_splits(seed: int, val_ratio: float) -> Tuple[HFDataset, HFDataset, HFDataset]:
    dd = load_dataset("openai/gsm8k", "main")
    train_full = dd["train"]
    test = dd["test"]
    train, val = split_train_val(train_full, val_ratio=val_ratio, seed=seed)
    return train, val, test


def load_svamp_splits(seed: int, val_ratio: float, test_ratio: float) -> Tuple[HFDataset, HFDataset, HFDataset]:
    dd = load_dataset("ChilleD/SVAMP")

    if isinstance(dd, DatasetDict) and "train" in dd and "test" in dd:
        train_full = dd["train"]
        test = dd["test"]
        train, val = split_train_val(train_full, val_ratio=val_ratio, seed=seed)
        return train, val, test

    if isinstance(dd, DatasetDict):
        split_name = "train" if "train" in dd else list(dd.keys())[0]
        full = dd[split_name]
    else:
        full = dd

    first = full.train_test_split(test_size=test_ratio, seed=seed)
    train_val = first["train"]
    test = first["test"]
    train, val = split_train_val(train_val, val_ratio=val_ratio, seed=seed)
    return train, val, test


def build_datasets(args) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    gsm_train, gsm_val, gsm_test = load_gsm8k_splits(args.seed, args.val_ratio)
    sv_train, sv_val, sv_test = load_svamp_splits(args.seed, args.val_ratio, args.svamp_test_ratio)

    train_parts: List[Dataset] = []
    val_parts: List[Dataset] = []

    if args.use_gsm8k:
        train_parts.append(MathPromptDataset(gsm_train, "gsm8k", args.target_mode, args.max_train_gsm8k, split_name="train"))
        val_parts.append(MathPromptDataset(gsm_val, "gsm8k", args.target_mode, args.max_val_gsm8k, split_name="validation"))
        gsm_test_ds = MathPromptDataset(gsm_test, "gsm8k", args.target_mode, args.max_test_gsm8k, split_name="test")
    else:
        gsm_test_ds = MathPromptDataset(gsm_test.select(range(0)), "gsm8k", args.target_mode, split_name="test")

    if args.use_svamp:
        train_parts.append(MathPromptDataset(sv_train, "svamp", args.target_mode, args.max_train_svamp, split_name="train"))
        val_parts.append(MathPromptDataset(sv_val, "svamp", args.target_mode, args.max_val_svamp, split_name="validation"))
        svamp_test_ds = MathPromptDataset(sv_test, "svamp", args.target_mode, args.max_test_svamp, split_name="test")
    else:
        svamp_test_ds = MathPromptDataset(sv_test.select(range(0)), "svamp", args.target_mode, split_name="test")

    if not train_parts:
        raise ValueError("At least one training dataset must be enabled.")

    train_ds = ConcatDataset(train_parts)
    val_ds = ConcatDataset(val_parts)
    return train_ds, val_ds, gsm_test_ds, svamp_test_ds


# -----------------------------------------------------------------------------
# Model loading / saving
# -----------------------------------------------------------------------------

def resolve_model_type(args) -> str:
    if args.model_type != "auto":
        return args.model_type

    ckpt_path = Path(args.checkpoint_path)
    config_path = ckpt_path / "config.json" if ckpt_path.is_dir() else ckpt_path.parent / "config.json"

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_type = str(json.load(f).get("model_type", "")).lower()
            if model_type == "diffusion_gemma" or "diffusion_gemma" in model_type:
                return "diffusion_gemma"
            if model_type == "mamba2" or "mamba2" in model_type:
                return "mamba2"
            if "gemma" in model_type:
                return "gemma4_text"
            if "gpt2" in model_type or "gpt" in model_type:
                return "gpt2"
        except Exception:
            pass

    if ckpt_path.suffix.lower() == ".safetensors":
        # Backward-compatible default for old GPT-2 safetensors checkpoints.
        return "gpt2"
    return "trlm"


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # Do not silently grow the vocabulary for HF checkpoints. DiffusionGemma and
    # causal baselines must use exactly the tokenizer vocabulary used in pretraining.
    if args.model_type_resolved == "trlm" and tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define either pad_token or eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    if is_mamba2_type(args.model_type_resolved):
        # Mamba2 masks padding states but its recurrent dynamics are primarily
        # validated with left padding; right padding may pollute later states.
        tokenizer.padding_side = "left"
    return tokenizer


def load_checkpoint_dict(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt_path = Path(path)
    if ckpt_path.suffix == ".safetensors":
        if safetensors_load_file is None:
            raise ImportError("Install safetensors to load .safetensors checkpoints: pip install safetensors")
        return safetensors_load_file(str(ckpt_path), device="cpu"), {}

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"], obj
    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        return obj, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def strip_common_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrapper prefixes for TRLM/GPT-2 style checkpoints.

    WARNING: Do not use this blindly for HF Gemma-style models, because
    `model.` is a real submodule prefix there, not just a wrapper prefix.
    """
    prefixes = ["module.", "model.", "_orig_mod."]
    out = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        out[new_key] = value
    return out


def strip_wrapper_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip only DDP/torch.compile wrapper prefixes.

    Unlike strip_common_prefixes(), this intentionally keeps `model.` because
    HF Gemma/Gemma-like causal LM classes expect keys such as
    `model.embed_tokens.weight` and `model.layers.0...`.
    """
    prefixes = ["module.", "_orig_mod."]
    out = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        out[new_key] = value
    return out


def align_state_dict_keys_to_model(
    state: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map checkpoint keys to the exact keys expected by `model`.

    This handles the common mismatch where a saved Gemma-like checkpoint uses
    `embed_tokens.weight` / `layers.0...`, while the HF causal LM wrapper
    expects `model.embed_tokens.weight` / `model.layers.0...`. It also handles
    the reverse direction when needed. Keys are remapped only when the target
    key exists in model.state_dict() and the tensor shape matches.
    """
    expected = model.state_dict()
    aligned: Dict[str, torch.Tensor] = {}

    def unique(seq):
        seen = set()
        out = []
        for item in seq:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    for key, value in state.items():
        k = key
        candidates = [k]

        if k.startswith("model."):
            candidates.append(k[len("model."):])
        else:
            candidates.append("model." + k)

        if k.startswith("language_model."):
            candidates.append(k[len("language_model."):])
        else:
            candidates.append("language_model." + k)

        if k.startswith("language_model.model."):
            candidates.append(k[len("language_model."):])
            candidates.append(k[len("language_model.model."):])

        chosen = k
        for cand in unique(candidates):
            if cand in expected and tuple(expected[cand].shape) == tuple(value.shape):
                chosen = cand
                break

        aligned[chosen] = value

    return aligned


def infer_gpt2_config_from_state(
    state: Dict[str, torch.Tensor],
    tokenizer,
    args,
) -> GPT2Config:
    state = strip_common_prefixes(state)

    if "transformer.wte.weight" not in state:
        raise ValueError(
            "Could not infer GPT-2 config because 'transformer.wte.weight' was not found. "
            "If your checkpoint is not a HuggingFace GPT2LMHeadModel state dict, convert it first "
            "or provide a checkpoint directory with config.json."
        )

    vocab_size, n_embd = state["transformer.wte.weight"].shape
    if "transformer.wpe.weight" in state:
        n_positions = int(state["transformer.wpe.weight"].shape[0])
    else:
        n_positions = int(args.max_seq_len)

    layer_ids = []
    for key in state.keys():
        m = re.match(r"transformer\.h\.(\d+)\.", key)
        if m:
            layer_ids.append(int(m.group(1)))
    n_layer = max(layer_ids) + 1 if layer_ids else args.gpt2_n_layer

    if args.gpt2_n_head is not None:
        n_head = args.gpt2_n_head
    else:
        # This cannot be recovered uniquely from a standard GPT-2 state dict.
        # Choose a common divisor and print a warning.
        candidates = [16, 12, 8, 6, 4, 2, 1]
        n_head = next((h for h in candidates if n_embd % h == 0), 1)
        print(
            f"[WARN] --gpt2_n_head was not provided and no config.json was found. "
            f"Inferred n_head={n_head} from n_embd={n_embd}. "
            "Pass --gpt2_n_head explicitly if your pretraining used a different value."
        )

    if n_embd % n_head != 0:
        raise ValueError(f"Invalid GPT-2 head count: n_embd={n_embd} is not divisible by n_head={n_head}")

    return GPT2Config(
        vocab_size=int(vocab_size),
        n_positions=max(int(n_positions), int(args.max_seq_len)),
        n_ctx=max(int(n_positions), int(args.max_seq_len)),
        n_embd=int(n_embd),
        n_layer=int(n_layer),
        n_head=int(n_head),
        n_inner=args.gpt2_n_inner,
        activation_function=args.gpt2_activation,
        resid_pdrop=args.gpt2_dropout,
        embd_pdrop=args.gpt2_dropout,
        attn_pdrop=args.gpt2_dropout,
        layer_norm_epsilon=args.gpt2_layer_norm_epsilon,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def load_trlm_model(args, tokenizer) -> TRLM:
    state, metadata = load_checkpoint_dict(args.checkpoint_path)
    ckpt_config = metadata.get("config") or metadata.get("model_config")
    if ckpt_config is None:
        raise ValueError("TRLM checkpoint must contain 'config' or 'model_config'.")

    config = TRLMConfig(**ckpt_config)

    if int(config.vocab_size) != int(len(tokenizer)):
        raise ValueError(
            f"Tokenizer/model vocab mismatch: config.vocab_size={config.vocab_size}, len(tokenizer)={len(tokenizer)}."
        )

    config.mask_token_id = tokenizer.mask_token_id
    if hasattr(config, "max_seq_len"):
        config.max_seq_len = args.max_seq_len

    model = TRLM(config=config)
    missing, unexpected = model.load_state_dict(strip_common_prefixes(state), strict=False)
    print(f"Loaded TRLM checkpoint: {args.checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return model


def load_gpt2_model(args, tokenizer) -> GPT2LMHeadModel:
    state, _ = load_checkpoint_dict(args.checkpoint_path)
    state = strip_common_prefixes(state)

    checkpoint_dir = Path(args.checkpoint_path).parent
    config = None

    if args.gpt2_config_path:
        config = AutoConfig.from_pretrained(args.gpt2_config_path)
    elif (checkpoint_dir / "config.json").exists():
        config = AutoConfig.from_pretrained(str(checkpoint_dir))
    else:
        config = infer_gpt2_config_from_state(state, tokenizer, args)

    if not isinstance(config, GPT2Config):
        # AutoConfig may return a compatible GPT2Config subclass. Keep only GPT-2-compatible fields.
        config = GPT2Config(**config.to_dict())

    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id

    if int(config.vocab_size) != int(len(tokenizer)):
        msg = f"GPT-2 vocab mismatch: config/checkpoint vocab_size={config.vocab_size}, len(tokenizer)={len(tokenizer)}."
        if args.allow_resize_token_embeddings:
            print("[WARN] " + msg + " Resizing token embeddings after loading.")
        else:
            raise ValueError(msg + " Use the same tokenizer or pass --allow_resize_token_embeddings.")

    if int(getattr(config, "n_positions", args.max_seq_len)) < int(args.max_seq_len):
        print(
            f"[WARN] config.n_positions={config.n_positions} < max_seq_len={args.max_seq_len}. "
            "Generation/training longer than pretrained position embeddings may fail."
        )

    model = GPT2LMHeadModel(config)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if int(model.config.vocab_size) != int(len(tokenizer)) and args.allow_resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    print(f"Loaded GPT-2 checkpoint: {args.checkpoint_path}")
    print(f"GPT-2 config: layers={model.config.n_layer}, embd={model.config.n_embd}, heads={model.config.n_head}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"First missing keys: {missing[:10]}")
    if unexpected:
        print(f"First unexpected keys: {unexpected[:10]}")
    return model


def build_gemma4_text_config(args, tokenizer, state: Dict[str, torch.Tensor]):
    """Build a Gemma4-Text config when config.json is not available.

    This requires the local transformers installation to know the `gemma4_text`
    model type. In the environment used for pretraining, run_clm.py already
    used `--model_type gemma4_text`, so AutoConfig.for_model should work there.
    """
    state = strip_wrapper_prefixes(state)
    vocab_size = len(tokenizer)

    # Prefer the checkpoint embedding shape if present.
    for key in ("model.embed_tokens.weight", "embed_tokens.weight", "language_model.model.embed_tokens.weight"):
        if key in state:
            vocab_size = int(state[key].shape[0])
            break

    kwargs = dict(
        vocab_size=vocab_size,
        hidden_size=args.gemma_hidden_size,
        intermediate_size=args.gemma_intermediate_size,
        num_hidden_layers=args.gemma_num_hidden_layers,
        num_attention_heads=args.gemma_num_attention_heads,
        num_key_value_heads=args.gemma_num_key_value_heads,
        head_dim=args.gemma_head_dim,
        global_head_dim=args.gemma_global_head_dim,
        max_position_embeddings=args.max_seq_len,
        hidden_activation=args.gemma_hidden_activation,
        rms_norm_eps=args.gemma_rms_norm_eps,
        attention_bias=args.gemma_attention_bias,
        attention_dropout=args.gemma_attention_dropout,
        sliding_window=args.gemma_sliding_window,
        hidden_size_per_layer_input=args.gemma_hidden_size_per_layer_input,
        use_cache=False,
        tie_word_embeddings=args.gemma_tie_word_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
    )

    try:
        return AutoConfig.for_model(args.gemma_auto_model_type, **kwargs)
    except Exception as exc:
        raise ValueError(
            "Could not build Gemma4-Text config automatically. Provide --gemma_config_path "
            "or place config.json next to model.safetensors. Original error: " + str(exc)
        ) from exc


def load_gemma4_text_model(args, tokenizer):
    state, _ = load_checkpoint_dict(args.checkpoint_path)
    # For Gemma-like HF causal LM models, `model.` is a real submodule prefix.
    # Only strip DDP/compile wrappers here; do NOT strip `model.` blindly.
    state = strip_wrapper_prefixes(state)

    checkpoint_dir = Path(args.checkpoint_path).parent

    if args.gemma_config_path:
        config = AutoConfig.from_pretrained(
            args.gemma_config_path,
            trust_remote_code=args.trust_remote_code,
        )
    elif (checkpoint_dir / "config.json").exists():
        config = AutoConfig.from_pretrained(
            str(checkpoint_dir),
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = build_gemma4_text_config(args, tokenizer, state)

    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id
    if hasattr(config, "use_cache"):
        config.use_cache = False

    if int(config.vocab_size) != int(len(tokenizer)):
        msg = f"Gemma4-Text vocab mismatch: config/checkpoint vocab_size={config.vocab_size}, len(tokenizer)={len(tokenizer)}."
        if args.allow_resize_token_embeddings:
            print("[WARN] " + msg + " Resizing token embeddings after loading.")
        else:
            raise ValueError(msg + " Use the same tokenizer or pass --allow_resize_token_embeddings.")

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=args.trust_remote_code,
    )

    state = align_state_dict_keys_to_model(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if int(model.config.vocab_size) != int(len(tokenizer)) and args.allow_resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    print(f"Loaded Gemma4-Text checkpoint: {args.checkpoint_path}")
    print(
        "Gemma4-Text config: "
        f"layers={getattr(model.config, 'num_hidden_layers', 'NA')}, "
        f"hidden={getattr(model.config, 'hidden_size', 'NA')}, "
        f"heads={getattr(model.config, 'num_attention_heads', 'NA')}, "
        f"kv_heads={getattr(model.config, 'num_key_value_heads', 'NA')}, "
        f"intermediate={getattr(model.config, 'intermediate_size', 'NA')}"
    )
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"First missing keys: {missing[:10]}")
    if unexpected:
        print(f"First unexpected keys: {unexpected[:10]}")

    # Large missing/unexpected lists usually mean the pretrained checkpoint was
    # not loaded at all. Fail fast instead of silently fine-tuning a random model.
    allowed_missing_suffixes = ("rotary_emb.inv_freq",)
    serious_missing = [k for k in missing if not k.endswith(allowed_missing_suffixes)]
    if len(serious_missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(
            "Gemma4-Text checkpoint did not load cleanly. "
            f"Serious missing keys: {len(serious_missing)}, unexpected keys: {len(unexpected)}. "
            "This usually indicates a key-prefix or model-class mismatch."
        )

    return model



def _set_standard_hf_tokenizer_ids(config, tokenizer) -> None:
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id


def load_mamba2_model(args, tokenizer):
    """Load a pretrained Hugging Face Mamba2 causal-LM checkpoint."""
    try:
        from transformers import Mamba2ForCausalLM
    except ImportError as exc:
        raise ImportError(
            "Could not import Mamba2ForCausalLM. Install a Transformers build "
            "that includes Mamba2 support."
        ) from exc

    ckpt_path = Path(args.checkpoint_path)
    checkpoint_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    config_source = args.mamba2_config_path or str(checkpoint_dir)

    config = AutoConfig.from_pretrained(
        config_source,
        trust_remote_code=args.trust_remote_code,
    )
    if getattr(config, "model_type", None) != "mamba2":
        raise ValueError(
            f"Expected a mamba2 config, got model_type={getattr(config, 'model_type', None)!r}."
        )

    checkpoint_vocab = int(getattr(config, "vocab_size"))
    if checkpoint_vocab != len(tokenizer) and not args.allow_resize_token_embeddings:
        raise ValueError(
            f"Mamba2 vocab mismatch: checkpoint vocab_size={checkpoint_vocab}, "
            f"len(tokenizer)={len(tokenizer)}. Use the pretraining tokenizer or pass "
            "--allow_resize_token_embeddings."
        )

    hidden_size = int(getattr(config, "hidden_size"))
    expand = int(getattr(config, "expand"))
    num_heads = int(getattr(config, "num_heads"))
    head_dim = int(getattr(config, "head_dim"))
    if hidden_size * expand != num_heads * head_dim:
        raise ValueError(
            "Invalid Mamba2 checkpoint dimensions: "
            f"hidden_size*expand={hidden_size * expand}, "
            f"num_heads*head_dim={num_heads * head_dim}."
        )

    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id
    # Fine-tuning computes full-sequence loss and does not need recurrent cache.
    config.use_cache = False

    has_hf_weights = any(
        (checkpoint_dir / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )

    if has_hf_weights:
        model = Mamba2ForCausalLM.from_pretrained(
            str(checkpoint_dir),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        if ckpt_path.is_dir():
            raise FileNotFoundError(
                f"No Hugging Face model weights were found in {checkpoint_dir}."
            )
        state, _ = load_checkpoint_dict(args.checkpoint_path)
        state = strip_wrapper_prefixes(state)
        model = Mamba2ForCausalLM(config)
        state = align_state_dict_keys_to_model(state, model)
        missing, unexpected = model.load_state_dict(state, strict=False)

        allowed_missing = set()
        if bool(getattr(config, "tie_word_embeddings", False)):
            allowed_missing.add("lm_head.weight")
        serious_missing = [key for key in missing if key not in allowed_missing]
        print(f"Mamba2 missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
        if serious_missing:
            print(f"First missing keys: {serious_missing[:10]}")
        if unexpected:
            print(f"First unexpected keys: {unexpected[:10]}")
        if len(serious_missing) > 10 or len(unexpected) > 10:
            raise RuntimeError(
                "Mamba2 checkpoint did not load cleanly. This usually indicates "
                "a model-class, config, or state-dict prefix mismatch."
            )

    if checkpoint_vocab != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        model.tie_weights()

    _set_standard_hf_tokenizer_ids(model.config, tokenizer)
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            model.generation_config.bos_token_id = tokenizer.bos_token_id

    n_params = sum({parameter.data_ptr(): parameter.numel() for parameter in model.parameters()}.values())
    print(f"Loaded Mamba2 checkpoint: {checkpoint_dir}")
    print(
        "Mamba2 config: "
        f"layers={getattr(model.config, 'num_hidden_layers', 'NA')}, "
        f"hidden={getattr(model.config, 'hidden_size', 'NA')}, "
        f"expand={getattr(model.config, 'expand', 'NA')}, "
        f"heads={getattr(model.config, 'num_heads', 'NA')}, "
        f"head_dim={getattr(model.config, 'head_dim', 'NA')}, "
        f"state_size={getattr(model.config, 'state_size', 'NA')}, "
        f"groups={getattr(model.config, 'n_groups', 'NA')}, "
        f"params={n_params / 1_000_000:.2f}M"
    )
    return model


def patch_diffusion_gemma_text_only_encoder() -> None:
    """Allow Transformers 5.13.x DiffusionGemma to load a text-only checkpoint."""
    try:
        from transformers.models.diffusion_gemma import modeling_diffusion_gemma as dg_modeling
    except ImportError as exc:
        raise ImportError(
            "DiffusionGemma requires a Transformers build with DiffusionGemma support "
            "(transformers>=5.11.0)."
        ) from exc

    encoder_cls = dg_modeling.DiffusionGemmaEncoderModel
    if getattr(encoder_cls, "_trlm_text_only_constructor_patch", False):
        return

    def text_only_encoder_init(self, config):
        dg_modeling.DiffusionGemmaPreTrainedModel.__init__(self, config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = dg_modeling.DiffusionGemmaEncoderTextModel(config=config.text_config)
        self.vision_tower = None
        self.embed_vision = None
        self.post_init()

    encoder_cls.__init__ = text_only_encoder_init
    encoder_cls._trlm_text_only_constructor_patch = True


def patch_diffusion_gemma_generation_config_compat() -> None:
    """
    Make older/transition Transformers DiffusionGemma builds tolerate
    ``return_dict_in_generate`` in generation_config.json.

    Some DiffusionGemmaGenerationConfig revisions inherit the generic
    GenerationConfig serialization output but reject this otherwise harmless
    key in their custom ``__init__``.  ``from_pretrained`` loads weights first
    and then crashes while reading generation_config.json.  Swallowing only
    this compatibility key is safe because DiffusionGemma generation already
    defaults to returning its structured generation output.
    """
    try:
        from transformers.models.diffusion_gemma import generation_diffusion_gemma as dg_generation
    except ImportError as exc:
        raise ImportError(
            "Could not import DiffusionGemma generation utilities. "
            "Install a Transformers build with DiffusionGemma support."
        ) from exc

    generation_config_cls = dg_generation.DiffusionGemmaGenerationConfig
    if getattr(generation_config_cls, "_trlm_return_dict_compat_patch", False):
        return

    # Newer releases already accept this field. Test before monkey-patching so
    # we leave those versions untouched.
    try:
        generation_config_cls(return_dict_in_generate=True)
    except ValueError as exc:
        if "return_dict_in_generate" not in str(exc):
            raise
    else:
        generation_config_cls._trlm_return_dict_compat_patch = True
        return

    original_init = generation_config_cls.__init__

    def compatible_init(self, **kwargs):
        # Do not re-attach the attribute on old implementations. Their
        # generation code uses getattr(..., True), and omitting it also prevents
        # save_pretrained() from writing the incompatible key again.
        kwargs.pop("return_dict_in_generate", None)
        original_init(self, **kwargs)

    generation_config_cls.__init__ = compatible_init
    generation_config_cls._trlm_return_dict_compat_patch = True
    print(
        "[INFO] Applied DiffusionGemma generation-config compatibility patch "
        "for return_dict_in_generate."
    )


def _set_diffusion_tokenizer_ids(config, tokenizer) -> None:
    text_config = getattr(config, "text_config", config)
    text_config.vocab_size = len(tokenizer)
    text_config.pad_token_id = tokenizer.pad_token_id
    text_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        text_config.bos_token_id = tokenizer.bos_token_id
    if hasattr(config, "pad_token_id"):
        config.pad_token_id = tokenizer.pad_token_id
    if hasattr(config, "eos_token_id"):
        config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None and hasattr(config, "bos_token_id"):
        config.bos_token_id = tokenizer.bos_token_id


def load_diffusion_gemma_model(args, tokenizer):
    try:
        from transformers import DiffusionGemmaForBlockDiffusion
    except ImportError as exc:
        raise ImportError(
            "Could not import DiffusionGemmaForBlockDiffusion. Install transformers>=5.11.0."
        ) from exc

    ckpt_path = Path(args.checkpoint_path)
    checkpoint_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    config_source = args.diffusion_config_path or str(checkpoint_dir)
    config = AutoConfig.from_pretrained(
        config_source,
        trust_remote_code=args.trust_remote_code,
    )
    if getattr(config, "model_type", None) != "diffusion_gemma":
        raise ValueError(
            f"Expected a diffusion_gemma config, got model_type={getattr(config, 'model_type', None)!r}."
        )

    config.vision_config = None
    patch_diffusion_gemma_text_only_encoder()
    patch_diffusion_gemma_generation_config_compat()

    text_config = getattr(config, "text_config", config)
    checkpoint_vocab = int(getattr(text_config, "vocab_size"))
    if checkpoint_vocab != len(tokenizer) and not args.allow_resize_token_embeddings:
        raise ValueError(
            f"DiffusionGemma vocab mismatch: checkpoint vocab_size={checkpoint_vocab}, "
            f"len(tokenizer)={len(tokenizer)}. Use the pretraining tokenizer or pass "
            "--allow_resize_token_embeddings."
        )

    # A Trainer checkpoint directory and a model.safetensors file next to
    # config.json are both loaded through the standard HF directory loader.
    has_hf_weights = any(
        (checkpoint_dir / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )

    if has_hf_weights:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            str(checkpoint_dir),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        state, _ = load_checkpoint_dict(args.checkpoint_path)
        state = strip_wrapper_prefixes(state)
        model = DiffusionGemmaForBlockDiffusion(config)
        state = align_state_dict_keys_to_model(state, model)
        missing, unexpected = model.load_state_dict(state, strict=False)
        serious_missing = [k for k in missing if not k.endswith("rotary_emb.inv_freq")]
        print(f"DiffusionGemma missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
        if serious_missing:
            print(f"First missing keys: {serious_missing[:10]}")
        if unexpected:
            print(f"First unexpected keys: {unexpected[:10]}")
        if len(serious_missing) > 10 or len(unexpected) > 10:
            raise RuntimeError("DiffusionGemma checkpoint did not load cleanly.")

    if checkpoint_vocab != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        _set_diffusion_tokenizer_ids(model.config, tokenizer)
    else:
        _set_diffusion_tokenizer_ids(model.config, tokenizer)

    canvas_length = int(getattr(model.config, "canvas_length", args.diffusion_canvas_length))
    args.diffusion_canvas_length = canvas_length
    text_config = getattr(model.config, "text_config", model.config)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Loaded DiffusionGemma checkpoint: {checkpoint_dir}")
    print(
        "DiffusionGemma config: "
        f"layers={getattr(text_config, 'num_hidden_layers', 'NA')}, "
        f"hidden={getattr(text_config, 'hidden_size', 'NA')}, "
        f"heads={getattr(text_config, 'num_attention_heads', 'NA')}, "
        f"kv_heads={getattr(text_config, 'num_key_value_heads', 'NA')}, "
        f"intermediate={getattr(text_config, 'intermediate_size', 'NA')}, "
        f"canvas_length={canvas_length}, params={n_params / 1_000_000:.2f}M"
    )
    return model

def load_model(args, tokenizer):
    if args.model_type_resolved == "trlm":
        return load_trlm_model(args, tokenizer)
    if args.model_type_resolved == "gpt2":
        return load_gpt2_model(args, tokenizer)
    if args.model_type_resolved == "gemma4_text":
        return load_gemma4_text_model(args, tokenizer)
    if args.model_type_resolved == "mamba2":
        return load_mamba2_model(args, tokenizer)
    if args.model_type_resolved == "diffusion_gemma":
        return load_diffusion_gemma_model(args, tokenizer)
    raise ValueError(f"Unsupported model_type: {args.model_type_resolved}")


def save_checkpoint(
    output_dir: str,
    name: str,
    model,
    tokenizer,
    args,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    if is_hf_pretrained_model_type(args.model_type_resolved):
        model.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
        train_state = {
            "global_step": step,
            "metrics": metrics or {},
            "args": vars(args),
        }
        if optimizer is not None:
            train_state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            train_state["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(train_state, out / "training_state.pt")
    else:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": config_to_dict(model.config),
            "global_step": step,
            "metrics": metrics or {},
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(ckpt, out / "model.pt")
        tokenizer.save_pretrained(out)

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

def make_steps(n: int) -> List[float]:
    n = max(1, int(n))
    return [1.0 / n] * n


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _diffusion_rand(
    shape,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=torch.float32, generator=generator)


def diffusion_gemma_loss(
    model,
    batch: Dict[str, Any],
    args,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    clean_canvas = batch["clean_canvas"].long()
    labels = batch["labels"].long()
    target_mask = labels.ne(-100)
    if not target_mask.any():
        raise ValueError("DiffusionGemma batch contains no supervised answer tokens.")

    batch_size, canvas_length = clean_canvas.shape
    if canvas_length != int(args.diffusion_canvas_length):
        raise ValueError(
            f"Canvas mismatch: batch={canvas_length}, config={args.diffusion_canvas_length}."
        )

    if model.training:
        noise_probability = _diffusion_rand(
            (batch_size, 1), device=clean_canvas.device, generator=generator
        )
        noise_probability = (
            args.diffusion_min_noise
            + noise_probability * (args.diffusion_max_noise - args.diffusion_min_noise)
        )
    else:
        noise_probability = torch.full(
            (batch_size, 1),
            float(args.diffusion_eval_noise),
            device=clean_canvas.device,
            dtype=torch.float32,
        )

    corruption_mask = (
        _diffusion_rand(
            clean_canvas.shape,
            device=clean_canvas.device,
            generator=generator,
        ) < noise_probability
    ) & target_mask

    # Guarantee at least one corrupted target per sequence.
    empty_rows = target_mask.any(dim=1) & ~corruption_mask.any(dim=1)
    if empty_rows.any():
        first_target = target_mask.to(torch.int64).argmax(dim=1)
        row_ids = empty_rows.nonzero(as_tuple=False).squeeze(1)
        corruption_mask[row_ids, first_target[row_ids]] = True

    text_config = getattr(model.config, "text_config", model.config)
    vocab_size = int(text_config.vocab_size)
    random_tokens = torch.randint(
        0,
        vocab_size,
        clean_canvas.shape,
        device=clean_canvas.device,
        generator=generator,
    )
    # Generation starts from a random full canvas. Keep unsupervised positions
    # random during SFT as well, so the answer cannot rely on a clean PAD tail.
    noisy_canvas = torch.where(target_mask, clean_canvas, random_tokens)
    noisy_canvas = torch.where(corruption_mask, random_tokens, noisy_canvas)

    # DiffusionGemma's decoder mask can reach PyTorch SDPA directly.
    # SDPA accepts a boolean mask or an additive floating-point mask, but not
    # the int64 0/1 mask produced by the collator. Keep 1/0 semantics by using
    # bool (True = attend, False = masked). Do NOT merely cast 0/1 to float,
    # because a floating-point SDPA mask is interpreted as an additive bias.
    encoder_attention_mask = batch["attention_mask"].to(dtype=torch.bool)
    canvas_attention_mask = torch.ones(
        clean_canvas.shape,
        dtype=torch.bool,
        device=clean_canvas.device,
    )
    decoder_attention_mask = torch.cat(
        [encoder_attention_mask, canvas_attention_mask], dim=1
    )
    model_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": encoder_attention_mask,
        "decoder_input_ids": noisy_canvas,
        "decoder_attention_mask": decoder_attention_mask,
    }

    if model.training and args.diffusion_self_conditioning_probability > 0.0:
        sc_mask = _diffusion_rand(
            (batch_size,), device=clean_canvas.device, generator=generator
        ) < args.diffusion_self_conditioning_probability
        if sc_mask.any():
            with torch.no_grad():
                first_outputs = model(**model_kwargs)
            model_kwargs["self_conditioning_logits"] = first_outputs.logits.detach()
            model_kwargs["self_conditioning_mask"] = sc_mask

    outputs = model(**model_kwargs)
    logits = outputs.logits
    if logits.shape[:2] != clean_canvas.shape:
        raise ValueError(
            f"DiffusionGemma logits shape {tuple(logits.shape)} does not match "
            f"canvas {tuple(clean_canvas.shape)}."
        )

    safe_targets = labels.masked_fill(~target_mask, 0)
    per_token_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).view_as(clean_canvas)

    weights = corruption_mask.float() / noise_probability.clamp_min(1e-4)
    loss = (per_token_loss * weights).sum() / weights.sum().clamp_min(1.0)

    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        masked_correct = ((predictions == safe_targets) & corruption_mask).sum()
        masked_total = corruption_mask.sum().clamp_min(1)
        masked_accuracy = masked_correct.float() / masked_total.float()
        mean_noise = noise_probability.mean()

    return loss, {
        "lm_loss": float(loss.detach().item()),
        "short_lm": 0.0,
        "consistency": 0.0,
        "depth": float(args.diffusion_gen_steps),
        "masked_accuracy": float(masked_accuracy.item()),
        "noise_probability": float(mean_noise.item()),
    }


def forward_loss(
    model,
    batch: Dict[str, Any],
    args,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if args.model_type_resolved == "trlm":
        if args.use_short_consistency:
            long_steps = make_steps(args.recursion_steps)
            short_n = random.randint(1, max(1, args.recursion_steps - 1))
            short_steps = make_steps(short_n)
        else:
            long_steps = make_steps(args.recursion_steps)
            short_steps = None

        _, loss, lm_loss_long, lm_loss_short, consistency_loss, depth, _ = model(
            masked_input_ids=batch["input_ids"],
            labels=batch["labels"],
            long_steps=long_steps,
            short_steps=short_steps,
        )
        return loss, {
            "lm_loss": float(lm_loss_long.item()) if torch.is_tensor(lm_loss_long) else float(loss.item()),
            "short_lm": float(lm_loss_short.item()) if torch.is_tensor(lm_loss_short) else 0.0,
            "consistency": float(consistency_loss.item()) if torch.is_tensor(consistency_loss) else 0.0,
            "depth": float(depth),
        }

    if is_diffusion_lm_type(args.model_type_resolved):
        return diffusion_gemma_loss(model, batch, args, generator=generator)

    model_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }
    if is_mamba2_type(args.model_type_resolved):
        # Full-sequence SFT does not need the recurrent generation cache.
        model_kwargs["use_cache"] = False
    outputs = model(**model_kwargs)
    return outputs.loss, {
        "lm_loss": float(outputs.loss.item()),
        "short_lm": 0.0,
        "consistency": 0.0,
        "depth": 0.0,
    }


def get_eval_answer_tokens(args) -> int:
    """Return the fixed answer canvas used by non-oracle generation evaluation."""
    configured = int(getattr(args, "eval_answer_tokens", -1))
    if configured > 0:
        return configured
    return int(args.max_answer_tokens)


def uses_oracle_answer_length(args) -> bool:
    return getattr(args, "eval_length_mode", "fixed") == "oracle"


def truncate_at_generation_stop(ids: torch.Tensor, tokenizer) -> List[int]:
    """Drop everything after the first generated EOS/PAD/MASK token."""
    stop_ids = {
        token_id
        for token_id in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "mask_token_id", None),
        )
        if token_id is not None
    }
    output: List[int] = []
    for token_id in ids.detach().cpu().tolist():
        token_id = int(token_id)
        if token_id in stop_ids:
            break
        output.append(token_id)
    return output


@torch.no_grad()
def generate_trlm(
    model: TRLM,
    batch: Dict[str, Any],
    tokenizer,
    args,
) -> torch.Tensor:
    """
    Generate a TRLM answer with a single latent-recursion pass.

    Fixed evaluation:
        Rebuilds every example as prompt + a fixed number of MASK tokens.

    Oracle evaluation:
        Uses the mask canvas already prepared by the collator and decodes only
        the gold answer length. Oracle mode should only be used for ablations.
    """

    if not uses_oracle_answer_length(args):
        input_ids = batch["input_ids"]
        prompt_lens = batch["prompt_lens"].detach().cpu().tolist()

        fixed_answer_tokens = get_eval_answer_tokens(args)
        max_prompt_len = max(
            1,
            int(args.max_seq_len) - fixed_answer_tokens,
        )

        rows: List[torch.Tensor] = []
        row_lengths: List[int] = []

        for row_index, prompt_len in enumerate(prompt_lens):
            # TRLM batches are right padded. Real prompt tokens begin at index 0.
            prompt = input_ids[row_index, :prompt_len]

            if prompt.numel() > max_prompt_len:
                prompt = prompt[-max_prompt_len:]

            mask_block = torch.full(
                (fixed_answer_tokens,),
                tokenizer.mask_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )

            row = torch.cat(
                [prompt, mask_block],
                dim=0,
            )

            rows.append(row)
            row_lengths.append(int(row.numel()))
            prompt_lens[row_index] = int(prompt.numel())

        batch_len = max(row_lengths)
        padded_rows: List[torch.Tensor] = []

        for row in rows:
            pad_len = batch_len - int(row.numel())

            if pad_len > 0:
                padding = torch.full(
                    (pad_len,),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                row = torch.cat([row, padding], dim=0)

            padded_rows.append(row)

        rebuilt_input_ids = torch.stack(
            padded_rows,
            dim=0,
        )

        # Do not mutate more batch fields than necessary.
        batch["input_ids"] = rebuilt_input_ids
        batch["masked_input_ids"] = rebuilt_input_ids
        batch["attention_mask"] = rebuilt_input_ids.ne(
            tokenizer.pad_token_id
        ).long()

        batch["prompt_lens"] = torch.tensor(
            prompt_lens,
            dtype=batch["prompt_lens"].dtype,
            device=batch["prompt_lens"].device,
        )

    forbidden_token_ids: List[int] = []

    # PAD should not occur inside the answer. Do not forbid it when PAD and EOS
    # share the same token ID, because that would also forbid EOS.
    if (
        tokenizer.pad_token_id is not None
        and tokenizer.pad_token_id != tokenizer.eos_token_id
    ):
        forbidden_token_ids.append(tokenizer.pad_token_id)

    # BOS is also not a valid answer-continuation token.
    if (
        tokenizer.bos_token_id is not None
        and tokenizer.bos_token_id != tokenizer.eos_token_id
        and tokenizer.bos_token_id != tokenizer.pad_token_id
    ):
        forbidden_token_ids.append(tokenizer.bos_token_id)

    return model.generate(
        masked_input_ids=batch["input_ids"],
        recursion_steps=args.recursion_steps,
        forbid_mask_token=True,
        forbid_token_ids=forbidden_token_ids,
        return_confidence=False,
    )


@torch.no_grad()
def generate_diffusion_gemma(model, batch: Dict[str, Any], tokenizer, args) -> torch.Tensor:
    try:
        from transformers import DiffusionGemmaGenerationConfig, EntropyBoundSamplerConfig
    except ImportError as exc:
        raise ImportError(
            "DiffusionGemma generation requires transformers>=5.11.0."
        ) from exc

    answer_lens = batch["answer_lens"].detach().cpu().tolist()
    if args.diffusion_max_new_tokens > 0:
        max_new_tokens = args.diffusion_max_new_tokens
    elif uses_oracle_answer_length(args) and args.diffusion_decode_gold_length:
        max_new_tokens = max(answer_lens) + args.diffusion_extra_new_tokens
    else:
        max_new_tokens = get_eval_answer_tokens(args)
    max_new_tokens = max(1, int(max_new_tokens))

    generation_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=max_new_tokens,
        max_denoising_steps=args.diffusion_gen_steps,
        sampler_config=EntropyBoundSamplerConfig(
            entropy_bound=args.diffusion_entropy_bound
        ),
        t_min=args.diffusion_t_min,
        t_max=args.diffusion_t_max,
        stability_threshold=args.diffusion_stability_threshold,
        confidence_threshold=args.diffusion_confidence_threshold,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = model.generate(
        input_ids=batch["input_ids"],
        # Keep the mask boolean so DiffusionGemma's decoder can pass it safely
        # to PyTorch SDPA during iterative denoising as well.
        attention_mask=batch["attention_mask"].to(dtype=torch.bool),
        generation_config=generation_config,
    )
    return generated.sequences if hasattr(generated, "sequences") else generated


@torch.no_grad()
def generate_causal_lm(model, batch: Dict[str, Any], tokenizer, args, device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"]
    prompt_lens = batch["prompt_lens"].detach().cpu().tolist()
    answer_lens = batch["answer_lens"].detach().cpu().tolist()

    prompt_rows = []
    prompt_masks = []
    max_prompt_len = max(prompt_lens)

    for i, plen in enumerate(prompt_lens):
        if is_mamba2_type(args.model_type_resolved):
            # Mamba2 training batches are left padded and contain prompt+answer.
            # Recover only the real prompt, excluding both left padding and the
            # teacher-forced answer tokens.
            answer_len = answer_lens[i]
            prompt_start = input_ids.shape[1] - (plen + answer_len)
            prompt = input_ids[i, prompt_start : prompt_start + plen]
        else:
            prompt = input_ids[i, :plen]
        pad_len = max_prompt_len - plen
        # Left padding is safer for decoder-only generation because the last
        # non-pad token is aligned at the same position for every sample.
        prompt_rows.append(torch.cat([
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long, device=device),
            prompt,
        ]))
        prompt_masks.append(torch.cat([
            torch.zeros((pad_len,), dtype=torch.long, device=device),
            torch.ones((plen,), dtype=torch.long, device=device),
        ]))

    prompt_batch = torch.stack(prompt_rows, dim=0)
    prompt_attention = torch.stack(prompt_masks, dim=0)

    if is_mamba2_type(args.model_type_resolved):
        fixed_max_new_tokens = args.mamba2_max_new_tokens
        extra_new_tokens = args.mamba2_extra_new_tokens
    else:
        fixed_max_new_tokens = args.gpt2_max_new_tokens
        extra_new_tokens = args.gpt2_extra_new_tokens

    if fixed_max_new_tokens > 0:
        max_new_tokens = fixed_max_new_tokens
    elif uses_oracle_answer_length(args):
        max_new_tokens = max(answer_lens) + extra_new_tokens
    else:
        max_new_tokens = get_eval_answer_tokens(args)

    generated = model.generate(
        input_ids=prompt_batch,
        attention_mask=prompt_attention,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # Return a tensor shaped like prompt+generated. The decode function below uses
    # prompt_len to slice the generated continuation.
    return generated


@torch.no_grad()
def generate_for_evaluation(
    model,
    batch: Dict[str, Any],
    tokenizer,
    args,
    device: torch.device,
    seed_offset: int = 0,
) -> torch.Tensor:
    """Dispatch generation while keeping DiffusionGemma evaluation reproducible."""
    if args.model_type_resolved == "trlm":
        return generate_trlm(model, batch, tokenizer, args)

    if is_diffusion_lm_type(args.model_type_resolved):
        # Preserve the surrounding training/evaluation RNG stream.
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = None
        if device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(device)

        eval_seed = int(args.diffusion_eval_seed) + int(seed_offset)
        torch.manual_seed(eval_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(eval_seed)

        try:
            return generate_diffusion_gemma(model, batch, tokenizer, args)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)

    return generate_causal_lm(model, batch, tokenizer, args, device)


def decode_generated_answer(
    generated: torch.Tensor,
    batch: Dict[str, Any],
    row_index: int,
    tokenizer,
    args,
) -> str:
    """Decode only the generated answer portion for one row."""
    prompt_lens = batch["prompt_lens"].detach().cpu().tolist()
    answer_lens = batch["answer_lens"].detach().cpu().tolist()

    if args.model_type_resolved == "trlm":
        start = prompt_lens[row_index]
        if uses_oracle_answer_length(args):
            end = start + answer_lens[row_index]
        else:
            end = start + get_eval_answer_tokens(args)
        pred_ids = generated[row_index, start:end]
    elif is_diffusion_lm_type(args.model_type_resolved):
        # DiffusionGemma returns left-padded prompt + denoised continuation.
        start = batch["input_ids"].shape[1]
        if uses_oracle_answer_length(args) and args.diffusion_decode_gold_length:
            end = start + answer_lens[row_index]
        else:
            end = generated.size(1)
        pred_ids = generated[row_index, start:end]
    else:
        # Causal generation returns left-padded prompt + continuation.
        start = max(prompt_lens)
        decode_gold_length = (
            args.mamba2_decode_gold_length
            if is_mamba2_type(args.model_type_resolved)
            else args.gpt2_decode_gold_length
        )
        if uses_oracle_answer_length(args) and decode_gold_length:
            end = start + answer_lens[row_index]
        else:
            end = generated.size(1)
        pred_ids = generated[row_index, start:end]

    return tokenizer.decode(
        truncate_at_generation_stop(pred_ids, tokenizer),
        skip_special_tokens=True,
    )


@torch.no_grad()
def evaluate_generation(
    model,
    loader: DataLoader,
    tokenizer,
    args,
    device: torch.device,
    name: str,
    max_batches: Optional[int] = None,
    save_predictions_path: Optional[str] = None,
) -> Dict[str, Any]:
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    loss_batches = 0
    predictions: List[Dict[str, Any]] = []

    # Test evaluations pass a predictions path. Generate those examples one at
    # a time so each answer has a real, independently measured latency rather
    # than an imprecise batch-time estimate.
    measure_per_sample_latency = save_predictions_path is not None
    generation_times: List[float] = []
    total_generation_wall_time = 0.0
    dataset_wall_start = time.perf_counter()
    global_example_index = 0

    model_info = get_model_parameter_stats(model, args.model_type_resolved)

    autocast_dtype = get_autocast_dtype(args.mixed_precision)
    use_autocast = autocast_dtype is not None and device.type == "cuda"

    diffusion_loss_generator = None
    if is_diffusion_lm_type(args.model_type_resolved):
        diffusion_loss_generator = torch.Generator(device=device)
        diffusion_loss_generator.manual_seed(args.diffusion_eval_seed)

    pbar = tqdm(loader, desc=f"eval:{name}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            loss, _ = forward_loss(
                model,
                batch,
                args,
                generator=diffusion_loss_generator,
            )
        total_loss += float(loss.item())
        loss_batches += 1

        batch_size = int(batch["input_ids"].shape[0])

        if measure_per_sample_latency:
            generation_records: List[Tuple[torch.Tensor, Dict[str, Any], int, float]] = []
            for row_index in range(batch_size):
                single_batch = slice_batch_item(batch, row_index)

                synchronize_device(device)
                generation_start = time.perf_counter()
                generated_device = generate_for_evaluation(
                    model,
                    single_batch,
                    tokenizer,
                    args,
                    device,
                    seed_offset=global_example_index + row_index,
                )
                synchronize_device(device)
                generation_seconds = time.perf_counter() - generation_start

                total_generation_wall_time += generation_seconds
                generation_times.append(generation_seconds)
                generation_records.append(
                    (generated_device.detach().cpu(), single_batch, 0, generation_seconds)
                )
        else:
            synchronize_device(device)
            generation_start = time.perf_counter()
            generated_device = generate_for_evaluation(
                model,
                batch,
                tokenizer,
                args,
                device,
                seed_offset=batch_idx,
            )
            synchronize_device(device)
            batch_generation_seconds = time.perf_counter() - generation_start
            total_generation_wall_time += batch_generation_seconds

            generated_cpu = generated_device.detach().cpu()
            generation_records = [
                (generated_cpu, batch, row_index, float("nan"))
                for row_index in range(batch_size)
            ]

        for generated, example_batch, decode_row_index, generation_seconds in generation_records:
            pred_text = decode_generated_answer(
                generated,
                example_batch,
                row_index=decode_row_index,
                tokenizer=tokenizer,
                args=args,
            )
            pred_num = extract_number(pred_text)
            gold_final = example_batch["gold_finals"][decode_row_index]
            gold_num = extract_number(gold_final)
            is_correct = numeric_equal(pred_num, gold_num, tol=args.answer_tolerance)

            total += 1
            correct += int(is_correct)

            should_save = (
                save_predictions_path is not None
                and (
                    args.max_saved_predictions < 0
                    or len(predictions) < args.max_saved_predictions
                )
            )
            if should_save:
                predictions.append({
                    "model_type": args.model_type_resolved,
                    "model_class": model_info["model_class"],
                    "eval_length_mode": getattr(args, "eval_length_mode", "fixed"),
                    "eval_answer_tokens": get_eval_answer_tokens(args),
                    "gold_answer_tokens": int(example_batch["answer_lens"][decode_row_index].detach().cpu().item()),
                    "model_total_parameters": model_info["total_parameters"],
                    "model_trainable_parameters": model_info["trainable_parameters"],
                    "example_id": example_batch["example_ids"][decode_row_index],
                    "dataset_split": example_batch["dataset_splits"][decode_row_index],
                    "dataset_index": example_batch["dataset_indices"][decode_row_index],
                    "source": example_batch["sources"][decode_row_index],
                    "question": example_batch["questions"][decode_row_index],
                    "prediction_text": pred_text,
                    "prediction_number": pred_num,
                    "gold": gold_final,
                    "gold_number": gold_num,
                    "correct": is_correct,
                    "generation_time_seconds": generation_seconds,
                    "generation_time_ms": generation_seconds * 1000.0,
                })

        global_example_index += batch_size
        acc = correct / max(1, total)
        postfix = {
            "acc": f"{acc:.4f}",
            "loss": f"{total_loss / max(1, loss_batches):.4f}",
        }
        if measure_per_sample_latency and generation_times:
            postfix["avg_ms"] = f"{np.mean(generation_times) * 1000.0:.2f}"
        pbar.set_postfix(postfix)

    synchronize_device(device)
    dataset_wall_time = time.perf_counter() - dataset_wall_start

    if generation_times:
        latency_array = np.asarray(generation_times, dtype=np.float64)
        average_generation_time = float(latency_array.mean())
        median_generation_time = float(np.median(latency_array))
        p95_generation_time = float(np.percentile(latency_array, 95))
        min_generation_time = float(latency_array.min())
        max_generation_time = float(latency_array.max())
    else:
        # Validation remains batched for speed, so individual latency
        # statistics are intentionally unavailable there.
        average_generation_time = (
            float(total_generation_wall_time / total) if total > 0 else 0.0
        )
        median_generation_time = None
        p95_generation_time = None
        min_generation_time = None
        max_generation_time = None

    metrics = {
        "name": name,
        "loss": total_loss / max(1, loss_batches),
        "accuracy": correct / max(1, total),
        "correct": correct,
        "total": total,
        "timing_mode": (
            "per_sample_sequential"
            if measure_per_sample_latency
            else "batched_generation"
        ),
        "dataset_wall_time_seconds": float(dataset_wall_time),
        "total_generation_time_seconds": float(total_generation_wall_time),
        "average_generation_time_seconds": average_generation_time,
        "average_generation_time_ms": average_generation_time * 1000.0,
        "median_generation_time_seconds": median_generation_time,
        "p95_generation_time_seconds": p95_generation_time,
        "min_generation_time_seconds": min_generation_time,
        "max_generation_time_seconds": max_generation_time,
        "evaluation_examples_per_second": (
            float(total / dataset_wall_time) if dataset_wall_time > 0 else 0.0
        ),
        "generation_examples_per_second": (
            float(total / total_generation_wall_time)
            if total_generation_wall_time > 0
            else 0.0
        ),
        "eval_length_mode": getattr(args, "eval_length_mode", "fixed"),
        "eval_answer_tokens": get_eval_answer_tokens(args),
        "model_total_parameters": model_info["total_parameters"],
        "model_trainable_parameters": model_info["trainable_parameters"],
        "model_total_parameters_millions": model_info["total_parameters_millions"],
    }

    if save_predictions_path:
        Path(save_predictions_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_predictions_path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary_path = Path(save_predictions_path).with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_info": model_info,
                    "dataset_metrics": metrics,
                    "example_id_format": "<dataset>_<split>_<zero-padded-index>",
                    "predictions_file": str(Path(save_predictions_path).name),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(
        f"[{name}] total={total} accuracy={metrics['accuracy']:.4f} "
        f"dataset_wall={dataset_wall_time:.3f}s "
        f"generation_total={total_generation_wall_time:.3f}s "
        f"avg_answer={average_generation_time * 1000.0:.3f}ms "
        f"params={model_info['total_parameters_millions']:.2f}M"
    )
    return metrics


def train(args) -> None:
    seed_everything(args.seed)
    args.model_type_resolved = resolve_model_type(args)
    print(f"Resolved model_type: {args.model_type_resolved}")

    if not 0.0 < args.diffusion_min_noise <= args.diffusion_max_noise <= 1.0:
        raise ValueError("Expected 0 < diffusion_min_noise <= diffusion_max_noise <= 1.")
    if not 0.0 < args.diffusion_eval_noise <= 1.0:
        raise ValueError("diffusion_eval_noise must be in (0, 1].")
    if not 0.0 <= args.diffusion_self_conditioning_probability <= 1.0:
        raise ValueError("diffusion_self_conditioning_probability must be in [0, 1].")
    if not uses_oracle_answer_length(args) and get_eval_answer_tokens(args) >= args.max_seq_len:
        raise ValueError(
            "Fixed-length evaluation requires eval_answer_tokens < max_seq_len "
            f"(got eval_answer_tokens={get_eval_answer_tokens(args)}, max_seq_len={args.max_seq_len})."
        )

    token = os.environ.get(args.hf_token_env, None)
    if token:
        login(token=token)

    tokenizer = load_tokenizer(args)
    train_ds, val_ds, gsm_test_ds, svamp_test_ds = build_datasets(args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args, tokenizer).to(device)
    model_info = get_model_parameter_stats(model, args.model_type_resolved)
    print(
        "Model parameters: "
        f"total={model_info['total_parameters']:,} "
        f"({model_info['total_parameters_millions']:.2f}M), "
        f"trainable={model_info['trainable_parameters']:,} "
        f"({model_info['trainable_parameters_millions']:.2f}M), "
        f"parameter_memory={model_info['parameter_memory_mib']:.2f} MiB"
    )

    if is_diffusion_lm_type(args.model_type_resolved):
        model_canvas_length = int(getattr(model.config, "canvas_length", args.diffusion_canvas_length))
        if args.diffusion_canvas_length != model_canvas_length:
            print(
                f"[INFO] Using checkpoint canvas_length={model_canvas_length} instead of "
                f"CLI value {args.diffusion_canvas_length}."
            )
        args.diffusion_canvas_length = model_canvas_length

    collator = MathCollator(
        tokenizer=tokenizer,
        model_type=args.model_type_resolved,
        max_seq_len=args.max_seq_len,
        max_answer_tokens=args.max_answer_tokens,
        pad_to_max_seq_len=args.pad_to_max_seq_len,
        diffusion_canvas_length=args.diffusion_canvas_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    gsm_test_loader = DataLoader(
        gsm_test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    svamp_test_loader = DataLoader(
        svamp_test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    print(f"Train examples: {len(train_ds)}")
    print(f"Val examples:   {len(val_ds)}")
    print(f"GSM8K test:     {len(gsm_test_ds)}")
    print(f"SVAMP test:     {len(svamp_test_ds)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.epochs
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    with open(Path(args.output_dir) / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    autocast_dtype = get_autocast_dtype(args.mixed_precision)
    use_autocast = autocast_dtype is not None and device.type == "cuda"

    best_val_acc = -1.0
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}")

        for micro_step, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                loss, extra = forward_loss(model, batch, args)
                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()
            running_loss += float(loss.item())

            should_step = (micro_step + 1) % args.gradient_accumulation_steps == 0
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / max(1, args.logging_steps * args.gradient_accumulation_steps)
                    lr = scheduler.get_last_lr()[0]
                    pbar.write(
                        f"step={global_step} epoch={epoch + 1} "
                        f"loss={avg_loss:.4f} lm_loss={extra['lm_loss']:.4f} "
                        f"short_lm={extra['short_lm']:.4f} "
                        f"cons={extra['consistency']:.4f} "
                        f"depth={extra['depth']} "
                        f"masked_acc={extra.get('masked_accuracy', 0.0):.4f} "
                        f"noise={extra.get('noise_probability', 0.0):.3f} "
                        f"grad={float(grad_norm):.4f} lr={lr:.2e}"
                    )
                    running_loss = 0.0

                if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                    max_eval_batches = None if args.max_eval_batches is None or args.max_eval_batches < 0 else args.max_eval_batches

                    val_metrics = evaluate_generation(
                        model,
                        val_loader,
                        tokenizer,
                        args,
                        device,
                        name="val",
                        max_batches=max_eval_batches,
                    )
                    pbar.write(f"VAL: {val_metrics}")
                    # evaluate_generation() switches the module to eval mode.
                    model.train()

                    if val_metrics["accuracy"] > best_val_acc:
                        best_val_acc = val_metrics["accuracy"]
                        save_checkpoint(
                            args.output_dir,
                            "checkpoint_best",
                            model,
                            tokenizer,
                            args,
                            optimizer,
                            scheduler,
                            step=global_step,
                            metrics=val_metrics,
                        )

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    save_checkpoint(
                        args.output_dir,
                        f"checkpoint_step_{global_step}",
                        model,
                        tokenizer,
                        args,
                        optimizer,
                        scheduler,
                        step=global_step,
                        metrics={"global_step": global_step},
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            pbar.set_postfix({
                "step": global_step,
                "loss": f"{float(loss.item()):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_checkpoint(
        args.output_dir,
        "checkpoint_final",
        model,
        tokenizer,
        args,
        optimizer,
        scheduler,
        step=global_step,
        metrics={"global_step": global_step},
    )

    final_metrics = {"model_info": model_info}
    if args.use_gsm8k and len(gsm_test_ds) > 0:
        final_metrics["gsm8k_test"] = evaluate_generation(
            model,
            gsm_test_loader,
            tokenizer,
            args,
            device,
            name="gsm8k_test",
            max_batches=None,
            save_predictions_path=str(Path(args.output_dir) / "gsm8k_test_predictions.jsonl"),
        )
    if args.use_svamp and len(svamp_test_ds) > 0:
        final_metrics["svamp_test"] = evaluate_generation(
            model,
            svamp_test_loader,
            tokenizer,
            args,
            device,
            name="svamp_test",
            max_batches=None,
            save_predictions_path=str(Path(args.output_dir) / "svamp_test_predictions.jsonl"),
        )

    with open(Path(args.output_dir) / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    print("\nFINAL TEST METRICS")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TRLM, GPT-2, Gemma4-Text, Mamba2, or DiffusionGemma on GSM8K + SVAMP.")

    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "trlm", "gpt2", "gemma4_text", "mamba2", "diffusion_gemma"])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="TRLM .pt or an HF GPT-2/Gemma4-Text/Mamba2/DiffusionGemma checkpoint directory/file")
    parser.add_argument("--tokenizer_name", type=str, default="DogaOytc/TRLM-tokenizer")
    parser.add_argument("--output_dir", type=str, default="checkpoints_math")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_answer_tokens", type=int, default=384)
    parser.add_argument(
        "--eval_length_mode",
        type=str,
        default="fixed",
        choices=["fixed", "oracle"],
        help=(
            "Generation evaluation length protocol. 'fixed' uses the same fixed "
            "answer canvas for every example; 'oracle' reproduces gold answer "
            "length decoding for ablation only."
        ),
    )
    parser.add_argument(
        "--eval_answer_tokens",
        type=int,
        default=-1,
        help=(
            "Fixed answer canvas for generation evaluation. Defaults to "
            "max_answer_tokens when <=0."
        ),
    )
    parser.add_argument("--target_mode", type=str, default="reasoning", choices=["reasoning", "final_only"])
    parser.add_argument("--pad_to_max_seq_len", action="store_true")

    parser.add_argument("--use_gsm8k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_svamp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--svamp_test_ratio", type=float, default=0.20)

    parser.add_argument("--max_train_gsm8k", type=int, default=None)
    parser.add_argument("--max_val_gsm8k", type=int, default=None)
    parser.add_argument("--max_test_gsm8k", type=int, default=None)
    parser.add_argument("--max_train_svamp", type=int, default=None)
    parser.add_argument("--max_val_svamp", type=int, default=None)
    parser.add_argument("--max_test_svamp", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=-1)

    # TRLM-specific inference/training knobs. Ignored in GPT-2 mode except kept in args for logging.
    parser.add_argument("--recursion_steps", type=int, default=8)
    parser.add_argument("--use_short_consistency", action="store_true")

    # GPT-2 loading/generation knobs.
    parser.add_argument("--gpt2_config_path", type=str, default=None, help="Optional GPT-2 config path or directory. Defaults to checkpoint dir/config.json if present.")
    parser.add_argument("--gpt2_n_head", type=int, default=None, help="Required for exact GPT-2 behavior if no config.json exists.")
    parser.add_argument("--gpt2_n_layer", type=int, default=6, help="Fallback only if n_layer cannot be inferred from checkpoint keys.")
    parser.add_argument("--gpt2_n_inner", type=int, default=None)
    parser.add_argument("--gpt2_activation", type=str, default="gelu_new")
    parser.add_argument("--gpt2_dropout", type=float, default=0.0)
    parser.add_argument("--gpt2_layer_norm_epsilon", type=float, default=1e-5)
    parser.add_argument("--gpt2_max_new_tokens", type=int, default=-1, help="If >0, override GPT-2 generation length. Otherwise uses fixed eval_answer_tokens, except in oracle eval mode.")
    parser.add_argument("--gpt2_extra_new_tokens", type=int, default=0)
    parser.add_argument("--gpt2_decode_gold_length", action=argparse.BooleanOptionalAction, default=True, help="Only applies when --eval_length_mode oracle.")
    parser.add_argument("--allow_resize_token_embeddings", action="store_true")

    # Mamba2 checkpoint loading and generation knobs.
    parser.add_argument("--mamba2_config_path", type=str, default=None, help="Optional Mamba2 config directory. Defaults to checkpoint directory.")
    parser.add_argument("--mamba2_max_new_tokens", type=int, default=-1, help="If >0, override Mamba2 generation length. Otherwise uses fixed eval_answer_tokens, except in oracle eval mode.")
    parser.add_argument("--mamba2_extra_new_tokens", type=int, default=0)
    parser.add_argument("--mamba2_decode_gold_length", action=argparse.BooleanOptionalAction, default=True, help="Only applies when --eval_length_mode oracle.")

    # Gemma4-Text loading knobs. These are only used when config.json is absent.
    parser.add_argument("--gemma_config_path", type=str, default=None, help="Optional Gemma4-Text config path or directory. Defaults to checkpoint dir/config.json if present.")
    parser.add_argument("--gemma_auto_model_type", type=str, default="gemma4_text", help="AutoConfig model_type to use when no config.json exists.")
    parser.add_argument("--gemma_hidden_size", type=int, default=512)
    parser.add_argument("--gemma_intermediate_size", type=int, default=2304)
    parser.add_argument("--gemma_num_hidden_layers", type=int, default=12)
    parser.add_argument("--gemma_num_attention_heads", type=int, default=8)
    parser.add_argument("--gemma_num_key_value_heads", type=int, default=4)
    parser.add_argument("--gemma_head_dim", type=int, default=64)
    parser.add_argument("--gemma_global_head_dim", type=int, default=64)
    parser.add_argument("--gemma_hidden_activation", type=str, default="gelu_pytorch_tanh")
    parser.add_argument("--gemma_rms_norm_eps", type=float, default=1e-6)
    parser.add_argument("--gemma_attention_bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gemma_attention_dropout", type=float, default=0.0)
    parser.add_argument("--gemma_sliding_window", type=int, default=512)
    parser.add_argument("--gemma_hidden_size_per_layer_input", type=int, default=0)
    parser.add_argument("--gemma_tie_word_embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)

    # DiffusionGemma loading, training objective, and generation knobs.
    parser.add_argument("--diffusion_config_path", type=str, default=None, help="Optional DiffusionGemma config directory. Defaults to checkpoint directory.")
    parser.add_argument("--diffusion_canvas_length", type=int, default=256, help="Fallback only; checkpoint config.canvas_length takes precedence.")
    parser.add_argument("--diffusion_min_noise", type=float, default=0.05)
    parser.add_argument("--diffusion_max_noise", type=float, default=1.0)
    parser.add_argument("--diffusion_eval_noise", type=float, default=0.5)
    parser.add_argument("--diffusion_self_conditioning_probability", type=float, default=0.0)
    parser.add_argument("--diffusion_gen_steps", type=int, default=48)
    parser.add_argument("--diffusion_entropy_bound", type=float, default=0.1)
    parser.add_argument("--diffusion_t_min", type=float, default=0.4)
    parser.add_argument("--diffusion_t_max", type=float, default=0.8)
    parser.add_argument("--diffusion_stability_threshold", type=int, default=1)
    parser.add_argument("--diffusion_confidence_threshold", type=float, default=0.005)
    parser.add_argument("--diffusion_max_new_tokens", type=int, default=-1, help="If >0, override DiffusionGemma generated length. Otherwise uses fixed eval_answer_tokens, except in oracle eval mode.")
    parser.add_argument("--diffusion_extra_new_tokens", type=int, default=0)
    parser.add_argument("--diffusion_decode_gold_length", action=argparse.BooleanOptionalAction, default=True, help="Only applies when --eval_length_mode oracle.")
    parser.add_argument("--diffusion_eval_seed", type=int, default=123456789)

    parser.add_argument("--answer_tolerance", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument(
        "--max_saved_predictions",
        type=int,
        default=-1,
        help="Maximum predictions to save per test set. Use -1 (default) to save every answer and its latency.",
    )

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "bf16", "fp16"])
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
