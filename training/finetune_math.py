#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune TRLM on GSM8K + SVAMP and evaluate on their test splits.

This script is written for your current TRLM interface:
    model(masked_input_ids=..., labels=..., long_steps=..., short_steps=...)

Training objective:
    Prompt tokens stay visible.
    Answer/solution tokens are replaced by [MASK].
    Loss is computed only on answer/solution tokens.

Important note for evaluation:
    TRLM's current generate() fills pre-existing mask positions only. Therefore,
    evaluation uses oracle answer length by masking exactly the gold target length.
    This is suitable for controlled masked-answer evaluation, not open-ended decoding.
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
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# -----------------------------------------------------------------------------
# Make project imports work when the script is placed under scripts/
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
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

    # Prefer explicit final-answer markers.
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

    # Fallback: use the last number in the text.
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

    # GSM8K answer format usually has reasoning + "#### final".
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


class MathMaskCollator:
    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 1024,
        max_answer_tokens: int = 384,
        pad_to_max_seq_len: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_answer_tokens = max_answer_tokens
        self.pad_to_max_seq_len = pad_to_max_seq_len

        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer must have a mask_token_id.")
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

            # Keep the final answer if an answer is too long.
            if len(answer_ids) > self.max_answer_tokens:
                answer_ids = answer_ids[-self.max_answer_tokens:]

            min_prompt_tokens = 8
            available_prompt_len = self.max_seq_len - len(answer_ids)
            if available_prompt_len < min_prompt_tokens:
                answer_ids = answer_ids[-(self.max_seq_len - min_prompt_tokens):]
                available_prompt_len = self.max_seq_len - len(answer_ids)

            if len(prompt_ids) > available_prompt_len:
                # Keep the end of the prompt because it contains the actual question.
                prompt_ids = prompt_ids[-available_prompt_len:]

            input_ids = prompt_ids + [self.tokenizer.mask_token_id] * len(answer_ids)
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

        return {
            "masked_input_ids": torch.tensor(input_batch, dtype=torch.long),
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

    # Some SVAMP repos expose one split only. If no official test split exists,
    # create a deterministic train/val/test split to avoid train-test leakage.
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

    train_ds = ConcatDataset(train_parts)
    val_ds = ConcatDataset(val_parts)
    return train_ds, val_ds, gsm_test_ds, svamp_test_ds


# -----------------------------------------------------------------------------
# Model loading / saving
# -----------------------------------------------------------------------------

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(args, tokenizer) -> TRLM:
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        ckpt_config = checkpoint.get("config") or checkpoint.get("model_config")
        if ckpt_config is None:
            raise ValueError("Checkpoint does not contain 'config' or 'model_config'.")
        config = TRLMConfig(**ckpt_config)
    else:
        config = TRLMConfig()
        config.vocab_size = len(tokenizer)
        config.mask_token_id = tokenizer.mask_token_id

    # Keep checkpoint vocab size. A vocab mismatch usually means the tokenizer used
    # for fine-tuning is not the tokenizer saved/used during pretraining.
    if int(config.vocab_size) != int(len(tokenizer)):
        raise ValueError(
            f"Tokenizer/model vocab mismatch: config.vocab_size={config.vocab_size}, "
            f"len(tokenizer)={len(tokenizer)}. Use the same tokenizer as pretraining, "
            "or resize/retrain embeddings explicitly."
        )

    config.mask_token_id = tokenizer.mask_token_id
    if hasattr(config, "max_seq_len"):
        config.max_seq_len = args.max_seq_len

    model = TRLM(config=config)

    if args.checkpoint_path:
        state = checkpoint["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint_path}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    return model


def save_checkpoint(
    output_dir: str,
    name: str,
    model: TRLM,
    tokenizer,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

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


@torch.no_grad()
def evaluate_generation(
    model: TRLM,
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
    predictions: List[Dict[str, Any]] = []

    autocast_dtype = get_autocast_dtype(args.mixed_precision)
    use_autocast = autocast_dtype is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=f"eval:{name}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        input_ids = batch["masked_input_ids"]
        labels = batch["labels"]

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            _, loss, *_ = model(
                masked_input_ids=input_ids,
                labels=labels,
                long_steps=make_steps(args.recursion_steps),
                short_steps=None,
            )
        total_loss += float(loss.item())

        generated = model.generate(
            masked_input_ids=input_ids,
            num_iterations=args.gen_iterations,
            recursion_steps=args.recursion_steps,
            temperature=0.0,
            top_k=None,
            do_sample=False,
            remask_ratio_start=args.remask_ratio_start,
            remask_ratio_end=0.0,
            forbid_mask_token=True,
        )

        generated = generated.detach().cpu()
        prompt_lens = batch["prompt_lens"].detach().cpu().tolist()
        answer_lens = batch["answer_lens"].detach().cpu().tolist()
        gold_finals = batch["gold_finals"]
        questions = batch["questions"]
        sources = batch["sources"]

        for i in range(generated.size(0)):
            start = prompt_lens[i]
            end = start + answer_lens[i]
            pred_text = tokenizer.decode(generated[i, start:end], skip_special_tokens=True)
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
        pbar.set_postfix({"acc": f"{acc:.4f}", "loss": f"{total_loss / (batch_idx + 1):.4f}"})

    metrics = {
        "name": name,
        "loss": total_loss / max(1, (batch_idx + 1 if 'batch_idx' in locals() else 1)),
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

    token = os.environ.get(args.hf_token_env, None)
    if token:
        login(token=token)

    tokenizer = load_tokenizer(args)
    train_ds, val_ds, gsm_test_ds, svamp_test_ds = build_datasets(args)

    collator = MathMaskCollator(
        tokenizer=tokenizer,
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
                if args.use_short_consistency:
                    # Teacher uses full recursion; student uses fewer recursion steps.
                    long_steps = make_steps(args.recursion_steps)
                    short_n = random.randint(1, max(1, args.recursion_steps - 1))
                    short_steps = make_steps(short_n)
                else:
                    long_steps = make_steps(args.recursion_steps)
                    short_steps = None

                _, loss, lm_loss_long, lm_loss_short, consistency_loss, depth, _ = model(
                    masked_input_ids=batch["masked_input_ids"],
                    labels=batch["labels"],
                    long_steps=long_steps,
                    short_steps=short_steps,
                )
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
                        f"loss={avg_loss:.4f} lm_loss={float(lm_loss_long.item()):.4f} "
                        f"short_lm={float(lm_loss_short.item()):.4f} "
                        f"cons={float(consistency_loss.item()):.4f} "
                        f"depth={depth} grad={float(grad_norm):.4f} lr={lr:.2e}"
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
        optimizer,
        scheduler,
        step=global_step,
        metrics={"global_step": global_step},
    )

    # Final held-out test evaluation.
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
    parser = argparse.ArgumentParser(description="Fine-tune TRLM on GSM8K + SVAMP.")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Pretrained TRLM checkpoint .pt path")
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

    parser.add_argument("--recursion_steps", type=int, default=8)
    parser.add_argument("--use_short_consistency", action="store_true")
    parser.add_argument("--gen_iterations", type=int, default=8)
    parser.add_argument("--remask_ratio_start", type=float, default=0.7)
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
