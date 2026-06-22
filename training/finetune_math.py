#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune TRLM or GPT-2 on GSM8K + SVAMP and evaluate on their test splits.

TRLM mode:
    Prompt tokens stay visible.
    Answer tokens are replaced by [MASK].
    Loss is computed only on answer tokens.
    Evaluation uses TRLM mask-filling generation.

GPT-2 mode:
    Prompt + answer tokens are fed autoregressively.
    Prompt labels are -100, answer labels are visible for causal LM training.
    Evaluation generates the answer from the prompt with greedy decoding.

This lets you compare TRLM and GPT-2 on the same tokenizer, same prompts,
same target formatting, same splits, and same numeric exact-match metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
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
    ) -> None:
        self.examples: List[Dict[str, str]] = []
        n = len(hf_dataset) if max_examples is None else min(len(hf_dataset), max_examples)

        for row in hf_dataset.select(range(n)):
            if source == "gsm8k":
                ex = format_gsm8k(row, target_mode=target_mode)
            elif source == "svamp":
                ex = format_svamp(row, target_mode=target_mode)
            else:
                raise ValueError(f"Unknown source: {source}")

            if extract_number(ex["gold_final"]) is not None:
                self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]


class MathCollator:
    def __init__(
        self,
        tokenizer,
        model_type: str,
        max_seq_len: int = 1024,
        max_answer_tokens: int = 384,
        pad_to_max_seq_len: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.max_answer_tokens = max_answer_tokens
        self.pad_to_max_seq_len = pad_to_max_seq_len

        if model_type == "trlm" and tokenizer.mask_token_id is None:
            raise ValueError("TRLM mode requires tokenizer.mask_token_id.")
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, Any]:
        encoded_rows = []

        for ex in features:
            prompt_ids = self._encode(ex["prompt"])
            answer_ids = self._encode(ex["target"])
            if self.tokenizer.eos_token_id is not None:
                answer_ids = answer_ids + [self.tokenizer.eos_token_id]

            if len(answer_ids) > self.max_answer_tokens:
                answer_ids = answer_ids[-self.max_answer_tokens:]

            min_prompt_tokens = 8
            available_prompt_len = self.max_seq_len - len(answer_ids)
            if available_prompt_len < min_prompt_tokens:
                answer_ids = answer_ids[-(self.max_seq_len - min_prompt_tokens):]
                available_prompt_len = self.max_seq_len - len(answer_ids)

            if len(prompt_ids) > available_prompt_len:
                prompt_ids = prompt_ids[-available_prompt_len:]

            if self.model_type == "trlm":
                input_ids = prompt_ids + [self.tokenizer.mask_token_id] * len(answer_ids)
            elif self.model_type == "gpt2":
                input_ids = prompt_ids + answer_ids
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

            labels = [-100] * len(prompt_ids) + answer_ids
            attention_mask = [1] * len(input_ids)

            encoded_rows.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "prompt_len": len(prompt_ids),
                "answer_len": len(answer_ids),
                "source": ex["source"],
                "gold_final": ex["gold_final"],
                "question": ex["question"],
                "target_text": ex["target"],
            })

        if self.pad_to_max_seq_len:
            batch_len = self.max_seq_len
        else:
            batch_len = min(self.max_seq_len, max(len(r["input_ids"]) for r in encoded_rows))

        input_batch, label_batch, mask_batch = [], [], []
        for row in encoded_rows:
            pad_len = batch_len - len(row["input_ids"])
            if pad_len < 0:
                row["input_ids"] = row["input_ids"][:batch_len]
                row["labels"] = row["labels"][:batch_len]
                row["attention_mask"] = row["attention_mask"][:batch_len]
                pad_len = 0

            input_batch.append(row["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            label_batch.append(row["labels"] + [-100] * pad_len)
            mask_batch.append(row["attention_mask"] + [0] * pad_len)

        input_tensor = torch.tensor(input_batch, dtype=torch.long)

        return {
            "input_ids": input_tensor,
            # Alias kept so old TRLM-specific code still works if needed.
            "masked_input_ids": input_tensor,
            "labels": torch.tensor(label_batch, dtype=torch.long),
            "attention_mask": torch.tensor(mask_batch, dtype=torch.long),
            "prompt_lens": torch.tensor([r["prompt_len"] for r in encoded_rows], dtype=torch.long),
            "answer_lens": torch.tensor([r["answer_len"] for r in encoded_rows], dtype=torch.long),
            "sources": [r["source"] for r in encoded_rows],
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
        train_parts.append(MathPromptDataset(gsm_train, "gsm8k", args.target_mode, args.max_train_gsm8k))
        val_parts.append(MathPromptDataset(gsm_val, "gsm8k", args.target_mode, args.max_val_gsm8k))
        gsm_test_ds = MathPromptDataset(gsm_test, "gsm8k", args.target_mode, args.max_test_gsm8k)
    else:
        gsm_test_ds = MathPromptDataset(gsm_test.select(range(0)), "gsm8k", args.target_mode)

    if args.use_svamp:
        train_parts.append(MathPromptDataset(sv_train, "svamp", args.target_mode, args.max_train_svamp))
        val_parts.append(MathPromptDataset(sv_val, "svamp", args.target_mode, args.max_val_svamp))
        svamp_test_ds = MathPromptDataset(sv_test, "svamp", args.target_mode, args.max_test_svamp)
    else:
        svamp_test_ds = MathPromptDataset(sv_test.select(range(0)), "svamp", args.target_mode)

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
    suffix = Path(args.checkpoint_path).suffix.lower()
    if suffix == ".safetensors":
        return "gpt2"
    return "trlm"


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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


def load_model(args, tokenizer):
    if args.model_type_resolved == "trlm":
        return load_trlm_model(args, tokenizer)
    if args.model_type_resolved == "gpt2":
        return load_gpt2_model(args, tokenizer)
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

    if args.model_type_resolved == "gpt2":
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


def forward_loss(model, batch: Dict[str, Any], args) -> Tuple[torch.Tensor, Dict[str, float]]:
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

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss, {
        "lm_loss": float(outputs.loss.item()),
        "short_lm": 0.0,
        "consistency": 0.0,
        "depth": 0.0,
    }


@torch.no_grad()
def generate_trlm(model: TRLM, batch: Dict[str, Any], tokenizer, args) -> torch.Tensor:
    return model.generate(
        masked_input_ids=batch["input_ids"],
        num_iterations=args.gen_iterations,
        recursion_steps=args.recursion_steps,
        temperature=0.0,
        top_k=None,
        do_sample=False,
        remask_ratio_start=args.remask_ratio_start,
        remask_ratio_end=0.0,
        forbid_mask_token=True,
    )


@torch.no_grad()
def generate_gpt2(model: GPT2LMHeadModel, batch: Dict[str, Any], tokenizer, args, device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"]
    prompt_lens = batch["prompt_lens"].detach().cpu().tolist()
    answer_lens = batch["answer_lens"].detach().cpu().tolist()

    prompt_rows = []
    prompt_masks = []
    max_prompt_len = max(prompt_lens)

    for i, plen in enumerate(prompt_lens):
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

    if args.gpt2_max_new_tokens > 0:
        max_new_tokens = args.gpt2_max_new_tokens
    else:
        # Comparable to TRLM's oracle target length, but generated autoregressively.
        max_new_tokens = max(answer_lens) + args.gpt2_extra_new_tokens

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

    autocast_dtype = get_autocast_dtype(args.mixed_precision)
    use_autocast = autocast_dtype is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=f"eval:{name}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            loss, _ = forward_loss(model, batch, args)
        total_loss += float(loss.item())
        loss_batches += 1

        if args.model_type_resolved == "trlm":
            generated = generate_trlm(model, batch, tokenizer, args).detach().cpu()
        else:
            generated = generate_gpt2(model, batch, tokenizer, args, device).detach().cpu()

        prompt_lens = batch["prompt_lens"].detach().cpu().tolist()
        answer_lens = batch["answer_lens"].detach().cpu().tolist()
        gold_finals = batch["gold_finals"]
        questions = batch["questions"]
        sources = batch["sources"]

        for i in range(generated.size(0)):
            if args.model_type_resolved == "trlm":
                start = prompt_lens[i]
                end = start + answer_lens[i]
                pred_ids = generated[i, start:end]
            else:
                # generated contains left-padded prompt + continuation.
                start = max(prompt_lens)
                if args.gpt2_decode_gold_length:
                    end = start + answer_lens[i]
                else:
                    end = generated.size(1)
                pred_ids = generated[i, start:end]

            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            pred_num = extract_number(pred_text)
            gold_num = extract_number(gold_finals[i])
            is_correct = numeric_equal(pred_num, gold_num, tol=args.answer_tolerance)

            total += 1
            correct += int(is_correct)

            if save_predictions_path and len(predictions) < args.max_saved_predictions:
                predictions.append({
                    "source": sources[i],
                    "question": questions[i],
                    "prediction_text": pred_text,
                    "prediction_number": pred_num,
                    "gold": gold_finals[i],
                    "gold_number": gold_num,
                    "correct": is_correct,
                })

        acc = correct / max(1, total)
        pbar.set_postfix({"acc": f"{acc:.4f}", "loss": f"{total_loss / max(1, loss_batches):.4f}"})

    metrics = {
        "name": name,
        "loss": total_loss / max(1, loss_batches),
        "accuracy": correct / max(1, total),
        "correct": correct,
        "total": total,
    }

    if save_predictions_path:
        Path(save_predictions_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_predictions_path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def train(args) -> None:
    seed_everything(args.seed)
    args.model_type_resolved = resolve_model_type(args)
    print(f"Resolved model_type: {args.model_type_resolved}")

    token = os.environ.get(args.hf_token_env, None)
    if token:
        login(token=token)

    tokenizer = load_tokenizer(args)
    train_ds, val_ds, gsm_test_ds, svamp_test_ds = build_datasets(args)

    collator = MathCollator(
        tokenizer=tokenizer,
        model_type=args.model_type_resolved,
        max_seq_len=args.max_seq_len,
        max_answer_tokens=args.max_answer_tokens,
        pad_to_max_seq_len=args.pad_to_max_seq_len,
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

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args, tokenizer).to(device)

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
                        f"depth={extra['depth']} grad={float(grad_norm):.4f} lr={lr:.2e}"
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

    final_metrics = {}
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
    parser = argparse.ArgumentParser(description="Fine-tune TRLM or GPT-2 on GSM8K + SVAMP.")

    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "trlm", "gpt2"])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="TRLM .pt or GPT-2 .safetensors/.pt checkpoint path")
    parser.add_argument("--tokenizer_name", type=str, default="DogaOytc/TRLM-tokenizer")
    parser.add_argument("--output_dir", type=str, default="checkpoints_math")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN")

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_answer_tokens", type=int, default=384)
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
    parser.add_argument("--gen_iterations", type=int, default=8)
    parser.add_argument("--remask_ratio_start", type=float, default=0.7)

    # GPT-2 loading/generation knobs.
    parser.add_argument("--gpt2_config_path", type=str, default=None, help="Optional GPT-2 config path or directory. Defaults to checkpoint dir/config.json if present.")
    parser.add_argument("--gpt2_n_head", type=int, default=None, help="Required for exact GPT-2 behavior if no config.json exists.")
    parser.add_argument("--gpt2_n_layer", type=int, default=6, help="Fallback only if n_layer cannot be inferred from checkpoint keys.")
    parser.add_argument("--gpt2_n_inner", type=int, default=None)
    parser.add_argument("--gpt2_activation", type=str, default="gelu_new")
    parser.add_argument("--gpt2_dropout", type=float, default=0.0)
    parser.add_argument("--gpt2_layer_norm_epsilon", type=float, default=1e-5)
    parser.add_argument("--gpt2_max_new_tokens", type=int, default=-1, help="If >0, fixed generation length for GPT-2. Otherwise uses max gold answer length in the batch.")
    parser.add_argument("--gpt2_extra_new_tokens", type=int, default=0)
    parser.add_argument("--gpt2_decode_gold_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_resize_token_embeddings", action="store_true")

    parser.add_argument("--answer_tolerance", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--max_saved_predictions", type=int, default=2000)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "bf16", "fp16"])
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
