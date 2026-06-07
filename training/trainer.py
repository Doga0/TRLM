import os
import shutil
import math
from itertools import cycle
from itertools import islice
from typing import Optional, Dict, Literal, Any, List, Tuple
from tqdm import tqdm
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tqdm.auto import tqdm
from safetensors.torch import save_file
from safetensors.torch import save_model

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import HfApi

from configs.config import TRLMConfig
from models.trlm import TRLM
from training.optimizer import MuonAdamW

from training.configuration_trlm import TRLMConfigHF

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters

    Maintains a shadow copy of model parameters that is updated as:
        ema_param = decay * ema_param + (1 - decay) * param

    Benefits for TRM-LLM:
    - Stabilizes recursive reasoning by smoothing parameter updates
    - The same reasoning network f(x,y,z) is applied n times recursively
    - EMA ensures consistent behavior across recursion depths
    - Improves generalization on validation/test sets

    Usage:
        ema = EMAModel(model, decay=0.9999)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update()  # Update EMA after each step

        # For evaluation, use EMA weights
        ema.apply_shadow()
        evaluate(model)
        ema.restore()  # Restore original weights for training
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: Model to track (can be DDP-wrapped or raw)
            decay: EMA decay rate (higher = slower updates, more smoothing)
                   Typical values: 0.999, 0.9999, 0.99999
            device: Device for EMA parameters (default: same as model)
        """
        self.decay = decay
        self.device = device

        # Get raw model (unwrap DDP if needed)
        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model

        # Create shadow parameters (EMA copy)
        self.shadow = {}
        self.backup = {}  # For storing original params during eval

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device:
                    self.shadow[name] = self.shadow[name].to(device)

    @torch.no_grad()
    def update(self):
        """Update EMA parameters after a training step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA update: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)

        Saves original weights to backup so they can be restored.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state dict for checkpointing"""
        return {
            'shadow': self.shadow.copy(),
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load EMA state from checkpoint"""
        self.decay = state_dict.get('decay', self.decay)

        # Load shadow tensors and move to correct device
        loaded_shadow = state_dict['shadow']
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in loaded_shadow:
                # Move shadow tensor to same device as model parameter
                self.shadow[name] = loaded_shadow[name].to(param.device)

    def copy_to_model(self):
        """Permanently copy EMA weights to model (for final model saving)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TRLMTrainer:
    def __init__(self,
                 model: TRLM,
                 train_dataloader,
                 config: TRLMConfig,
                 eval_dataloader = None
            ):
        
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.epochs = self.config.epochs
        self.warmup_steps = (
            self.config.warmup_steps
            if self.config.warmup_steps is not None
            else self.config.get_warmup_steps(self.config.max_steps_per_epoch)
        )
        
        self.device = next(self.model.parameters()).device

        self.logger = self._setup_logger()
        self.writer = self._setup_tensorboard()

        self.optimizer = self._setup_optimizer()
        
        self.scheduler = self._setup_scheduler()

        self.log_model_size()
        self.log_param_breakdown()

        ema_decay = getattr(config, "ema_decay", 0.9999)
        self.ema = EMAModel(self.model, decay=ema_decay, device=self.device)

        self.global_step = 0
        self.current_epoch = 0
        self.tokens_seen = 0
        self.best_val_loss = float('inf')
        self.training_history: List[Dict[str, float]] = []

        self.saved_checkpoints: List[str] = []

    def _estimate_tokens_per_micro_batch(self) -> int:
        batch_size = getattr(self.train_dataloader, "batch_size", None)

        if batch_size is None:
            batch_size = 1

        max_seq_len = getattr(self.config, "max_seq_len", 1024)

        return int(batch_size * max_seq_len)

    def _estimate_total_optimizer_steps(self) -> int:
        max_train_tokens = int(getattr(self.config, "max_train_tokens", 1_000_000_000))
        grad_accum = int(getattr(self.config, "gradient_accumulation_steps", 1))

        tokens_per_micro_batch = self._estimate_tokens_per_micro_batch()
        tokens_per_optimizer_step = max(1, tokens_per_micro_batch * grad_accum)

        return max(1, math.ceil(max_train_tokens / tokens_per_optimizer_step))

    def log_model_size(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.logger.info(f"Non-trainable Parameters: {total_params - trainable_params:,}")

    def log_param_breakdown(self):
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            self.logger.info(f"{name}: {params:,} params")

    def _setup_scheduler(self):
        total_training_steps = self._estimate_total_optimizer_steps()

        warmup_steps = getattr(self.config, "warmup_steps", None)

        if warmup_steps is None:
            warmup_ratio = float(getattr(self.config, "warmup_ratio", 0.03))
            warmup_steps = int(total_training_steps * warmup_ratio)

        self.logger.info(f"Estimated total optimizer steps: {total_training_steps:,}")
        self.logger.info(f"Warmup steps: {warmup_steps:,}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        return scheduler

    def _setup_optimizer(self):
        muon_lr = getattr(self.config, "muon_lr", 0.02)
        muon_momentum = getattr(self.config, "muon_momentum", 0.95)
        muon_beta2 = getattr(self.config, "muon_beta2", 0.95)
        muon_ns_steps = getattr(self.config, "muon_ns_steps", 5)
        
        adam_lr = getattr(self.config, "learning_rate", 3e-4)
        adam_betas = getattr(self.config, "adamw_betas", (0.9, 0.95))
        adam_eps = getattr(self.config, "adamw_eps", 1e-8)
        
        weight_decay = getattr(self.config, "weight_decay", 0.01)

        muon_params_by_shape = {}

        adamw_decay_params = []
        adamw_no_decay_params = []

        no_decay_keywords = ["bias", "norm", "ln_"]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and norm parameters (AdamW)
            if param.ndim < 2 or any(nd in name.lower() for nd in no_decay_keywords):
                adamw_no_decay_params.append(param)
            
            # Muon for 2D hidden weights inside the core network
            elif param.ndim == 2 and "reasoning_module" in name:
                shape = tuple(param.shape)
                if shape not in muon_params_by_shape:
                    muon_params_by_shape[shape] = []
                muon_params_by_shape[shape].append(param)
                
            # AdamW with weight decay for everything else (Embeddings, lm_head, q_head)
            else:
                adamw_decay_params.append(param)

        param_groups = []
        
        for shape, params in muon_params_by_shape.items():
            param_groups.append({
                "params": params,
                "kind": "muon",
                "lr": muon_lr,
                "momentum": muon_momentum,
                "beta2": muon_beta2,
                "ns_steps": muon_ns_steps,
                "weight_decay": weight_decay,
            })

        if adamw_decay_params:
            param_groups.append({
                "params": adamw_decay_params,
                "kind": "adamw", 
                "lr": adam_lr,
                "betas": adam_betas,
                "eps": adam_eps,
                "weight_decay": weight_decay
            })
            
        if adamw_no_decay_params:
            param_groups.append({
                "params": adamw_no_decay_params,
                "kind": "adamw", 
                "lr": adam_lr,
                "betas": adam_betas,
                "eps": adam_eps,
                "weight_decay": 0.0  
            })

        grouped_param_ids = set()
        duplicate_params = []

        for group in param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in grouped_param_ids:
                    duplicate_params.append(pid)
                grouped_param_ids.add(pid)

        missing_params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in grouped_param_ids:
                missing_params.append(name)

        if duplicate_params:
            raise RuntimeError(
                f"Some parameters appear in more than one optimizer group. "
                f"Duplicate count: {len(duplicate_params)}"
            )

        if missing_params:
            raise RuntimeError(
                "Some trainable parameters are missing from optimizer groups:\n"
                + "\n".join(missing_params[:100])
            )

        # Safety check: every Muon group must contain same-shaped tensors.
        for i, group in enumerate(param_groups):
            if group["kind"] == "muon":
                shapes = {tuple(p.shape) for p in group["params"]}
                if len(shapes) != 1:
                    raise RuntimeError(
                        f"Muon group {i} has mixed shapes: {shapes}"
                    )

        print("\nOptimizer parameter groups:")
        for i, group in enumerate(param_groups):
            n_params = sum(p.numel() for p in group["params"])
            n_tensors = len(group["params"])

            if group["kind"] == "muon":
                shape = tuple(group["params"][0].shape)
                print(
                    f"  Group {i}: MUON  | tensors={n_tensors} | "
                    f"shape={shape} | params={n_params:,} | lr={group['lr']}"
                )
            else:
                print(
                    f"  Group {i}: ADAMW | tensors={n_tensors} | "
                    f"params={n_params:,} | lr={group['lr']} | "
                    f"wd={group['weight_decay']}"
                )

        return MuonAdamW(param_groups)

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger."""

        TRAIN_LEVEL_NUM = 21
        EVAL_LEVEL_NUM = 22

        logging.addLevelName(TRAIN_LEVEL_NUM, "TRAIN")
        logging.addLevelName(EVAL_LEVEL_NUM, "EVAL")

        if not hasattr(logging.Logger, "train"):
            def log_train(self, message, *args, **kws):
                if self.isEnabledFor(TRAIN_LEVEL_NUM):
                    self._log(TRAIN_LEVEL_NUM, message, args, **kws)
            logging.Logger.train = log_train

        if not hasattr(logging.Logger, "eval"):
            def log_eval(self, message, *args, **kws):
                if self.isEnabledFor(EVAL_LEVEL_NUM):
                    self._log(EVAL_LEVEL_NUM, message, args, **kws)
            logging.Logger.eval = log_eval

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if logger.handlers:
            logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.propagate = False

        return logger

    def _setup_tensorboard(self) -> Optional[Any]:
        """Setup TensorBoard writer."""
        if not self.config.use_tensorboard or not TENSORBOARD_AVAILABLE:
            return None
        
        os.makedirs(self.config.log_dir, exist_ok=True)

        return SummaryWriter(log_dir=self.config.log_dir, flush_secs=5)

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all configured destinations."""

        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def sample_short_trajectory(self, max_loops):
        """
        Return a random discrete trajectory from 0 to 1, as an array of step sizes
        drawn from {x / max_loops | x=1..max_loops-1}, with total sum == 1.

        The number of steps (trajectory length) L is chosen uniformly from 1 to max_loops-1.
        """
        assert max_loops >= 2, "max_loops must be >= 2"

        allowed = [x / max_loops for x in range(1, max_loops+1)]  # possible step sizes
        min_step = allowed[0]

        # choose path length L uniformly
        L = random.randint(1, max_loops - 1)

        trajectory = []
        remaining = 1.0
        remaining_steps = L

        for _ in range(L - 1):
            # max we can take now so that (remaining_steps - 1) * min_step can still fit
            cap = remaining - (remaining_steps - 1) * min_step
            # pick any allowed step <= cap
            choices = [s for s in allowed if s <= cap + 1e-12]
            v = random.choice(choices)
            trajectory.append(v)
            remaining -= v
            remaining_steps -= 1

        # last step must exactly fill to 1
        trajectory.append(remaining)

        return trajectory
    
    def sample_teacher_student_trajectories(self):
        """
        Teacher/student recursion training.

        Teacher uses a variable number of uniform steps.
        Student uses your existing random short trajectory sampler.
        """

        max_steps = getattr(self.config, "max_recursions", 8)
        min_steps = getattr(self.config, "min_recursions", 1)

        max_steps = max(2, int(max_steps))
        min_steps = max(1, int(min_steps))

        teacher_steps = random.randint(
            max(2, min_steps),
            max_steps,
        )

        teacher_trajectory = [1.0 / teacher_steps] * teacher_steps

        student_trajectory = self.sample_short_trajectory(teacher_steps)

        return teacher_trajectory, student_trajectory

    def train_step(self, batch: Dict[str, torch.Tensor], micro_step_idx: int):
        """
        Executes a single training iteration: forward pass, backward pass, and 
        conditional optimizer step based on gradient accumulation.
        """

        batch = self._move_to_device(batch)
        masked_input_ids = batch["masked_input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            long_trajectory, short_trajectory = self.sample_teacher_student_trajectories()
            
            (
                logits,
                loss,
                lm_loss_long,
                lm_loss_short,
                consistency_loss,
                depth,
                _,
            ) = self.model(
                masked_input_ids=masked_input_ids,
                labels=labels,
                long_steps=long_trajectory,
                short_steps=short_trajectory,
            )

            scaled_loss = loss / self.config.gradient_accumulation_steps

        scaled_loss.backward()

        # Optimizer Step
        grad_norm = 0.0
        did_optimizer_step = False
        
        if (micro_step_idx + 1) % self.config.gradient_accumulation_steps == 0:
            
            # Gradient clipping
            max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm).item()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Update EMA 
            if self.ema is not None:
                self.ema.update()

            self.global_step += 1
            did_optimizer_step = True

        safe_lm_loss = torch.clamp(lm_loss_long, max=20.0)
        perplexity = torch.exp(safe_lm_loss).item()

        if attention_mask is not None:
            input_tokens = int(attention_mask.sum().item())
        else:
            input_tokens = int(masked_input_ids.numel())

        target_tokens = int(labels.ne(-100).sum().item())

        return {
            "loss": loss.item(),
            "lm_loss": lm_loss_long.item(),
            "short_lm_loss": lm_loss_short.item() if torch.is_tensor(lm_loss_short) else 0.0,
            "consistency_loss": consistency_loss.item() if torch.is_tensor(consistency_loss) else 0.0,
            "perplexity": perplexity,
            "recursion_depth": depth,
            "grad_norm": grad_norm,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "did_optimizer_step": did_optimizer_step,
        }

    def train_epoch(self):
        """
        Runs one training epoch/chunk.
        Handles progress tracking, logging, periodic evaluation, and checkpointing.

        Training still follows the existing architecture:
            train() -> train_epoch() -> train_step()

        The stopping criterion is now token-budget aware.
        """

        self.model.train()

        total_loss = 0.0
        total_lm_loss = 0.0
        local_micro_steps = 0

        max_train_tokens = int(getattr(self.config, "max_train_tokens", 1_000_000_000))
        max_steps_per_epoch = int(getattr(self.config, "max_steps_per_epoch", 100_000))

        remaining_tokens = max(0, max_train_tokens - self.tokens_seen)

        if remaining_tokens <= 0:
            self.logger.info("Token budget already reached. Skipping train_epoch().")
            return 0.0

        tokens_per_micro_batch = self._estimate_tokens_per_micro_batch()
        tokens_per_optimizer_step = tokens_per_micro_batch * self.config.gradient_accumulation_steps

        total_opt_steps = math.ceil(remaining_tokens / tokens_per_optimizer_step)

        pbar = tqdm(
            total=total_opt_steps,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            dynamic_ncols=True,
        )

        for step_idx, batch in enumerate(self.train_dataloader):
            if self.tokens_seen >= max_train_tokens:
                self.logger.info(
                    f"Reached token budget: {self.tokens_seen:,}/{max_train_tokens:,}"
                )
                break

            step_metrics = self.train_step(batch, step_idx)

            local_micro_steps += 1

            total_loss += step_metrics["loss"]
            total_lm_loss += step_metrics["lm_loss"]

            self.tokens_seen += step_metrics["input_tokens"]

            if not hasattr(self, "target_tokens_seen"):
                self.target_tokens_seen = 0

            self.target_tokens_seen += step_metrics["target_tokens"]

            avg_loss = total_loss / local_micro_steps
            avg_lm_loss = total_lm_loss / local_micro_steps
            ppl = step_metrics["perplexity"]

            pbar.set_postfix({
                "step": self.global_step,
                "tokens": f"{self.tokens_seen / 1e9:.3f}B",
                "targets": f"{self.target_tokens_seen / 1e9:.3f}B",
                "loss": f"{avg_loss:.4f}",
                "lm_loss": f"{avg_lm_loss:.4f}",
                "short_lm_loss": f"{step_metrics['short_lm_loss']:.4f}",
                "consistency_loss": f"{step_metrics['consistency_loss']:.4f}",
                "masked_ppl": f"{ppl:.4f}",
                "grad_norm": f"{step_metrics['grad_norm']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                "depth": f"{step_metrics['recursion_depth']}",
            })

            if step_metrics["did_optimizer_step"]:
                pbar.update(1)

                # ------------------------------------------------------------
                # Token-based logging
                # ------------------------------------------------------------
                should_log_by_tokens = (
                    hasattr(self, "next_log_tokens")
                    and self.tokens_seen >= self.next_log_tokens
                )

                should_log_by_steps = (
                    not hasattr(self, "next_log_tokens")
                    and self.global_step % getattr(self.config, "logging_steps", 10) == 0
                )

                if should_log_by_tokens or should_log_by_steps:
                    log_dict = {
                        "Train/Loss": avg_loss,
                        "Train/LmLoss": step_metrics["lm_loss"],
                        "Train/AvgLmLoss": avg_lm_loss,
                        "Train/ShortLmLoss": step_metrics["short_lm_loss"],
                        "Train/ConsistencyLoss": step_metrics["consistency_loss"],
                        "Train/MaskedPerplexity": ppl,
                        "Train/LearningRate": self.optimizer.param_groups[0]["lr"],
                        "Train/RecursionDepth": step_metrics["recursion_depth"],
                        "Train/TokensSeen": self.tokens_seen,
                        "Train/TargetTokensSeen": self.target_tokens_seen,
                        "Train/GradNorm": step_metrics["grad_norm"],
                    }

                    self._log_metrics(log_dict, step=self.global_step)

                    self.logger.train(
                        f"Step {self.global_step} | "
                        f"Tokens: {self.tokens_seen:,} | "
                        f"TargetTokens: {self.target_tokens_seen:,} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LmLoss: {step_metrics['lm_loss']:.4f} | "
                        f"AvgLmLoss: {avg_lm_loss:.4f} | "
                        f"MaskedPerplexity: {ppl:.2f} | "
                        f"GradNorm: {step_metrics['grad_norm']:.4f} | "
                        f"Depth: {step_metrics['recursion_depth']}"
                    )

                    if hasattr(self, "next_log_tokens"):
                        self.next_log_tokens += int(
                            getattr(self.config, "log_token_interval", 10_000_000)
                        )

                # ------------------------------------------------------------
                # Token-based evaluation
                # ------------------------------------------------------------
                should_eval_by_tokens = (
                    hasattr(self, "next_eval_tokens")
                    and self.tokens_seen >= self.next_eval_tokens
                )

                should_eval_by_steps = (
                    not hasattr(self, "next_eval_tokens")
                    and self.global_step % getattr(self.config, "eval_interval", 10) == 0
                )

                if should_eval_by_tokens or should_eval_by_steps:
                    (
                        avg_val_loss,
                        val_perplexity,
                        acc_long,
                        acc_short,
                        acc_diff,
                        consistency,
                        avg_depth,
                    ) = self.evaluate()

                    self.logger.eval(
                        f"Eval | "
                        f"Step {self.global_step} | "
                        f"Tokens {self.tokens_seen:,} | "
                        f"Loss: {avg_val_loss:.4f} | "
                        f"MaskedPerplexity: {val_perplexity:.2f} | "
                        f"AccLong: {acc_long:.4f} | "
                        f"AccShort: {acc_short:.4f} | "
                        f"AccDiff: {acc_diff:.4f} | "
                        f"Consistency: {consistency:.4f} | "
                        f"Depth: {avg_depth:.2f}"
                    )

                    self._log_metrics({
                        "Eval/Loss": avg_val_loss,
                        "Eval/MaskedPerplexity": val_perplexity,
                        "Eval/AccLong": acc_long,
                        "Eval/AccShort": acc_short,
                        "Eval/AccDiff": acc_diff,
                        "Eval/Consistency": consistency,
                        "Eval/Depth": avg_depth,
                        "Eval/TokensSeen": self.tokens_seen,
                        "Eval/TargetTokensSeen": self.target_tokens_seen,
                    }, step=self.global_step)

                    self.model.train()

                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.save_checkpoint(is_best=True)

                    if hasattr(self, "next_eval_tokens"):
                        self.next_eval_tokens += int(
                            getattr(self.config, "eval_token_interval", 250_000_000)
                        )

                # ------------------------------------------------------------
                # Token-based checkpointing
                # ------------------------------------------------------------
                should_save_by_tokens = (
                    hasattr(self, "next_save_tokens")
                    and self.tokens_seen >= self.next_save_tokens
                )

                should_save_by_steps = (
                    not hasattr(self, "next_save_tokens")
                    and self.global_step % getattr(self.config, "save_steps", 1000) == 0
                )

                if should_save_by_tokens or should_save_by_steps:
                    self.save_checkpoint()

                    if hasattr(self, "next_save_tokens"):
                        self.next_save_tokens += int(
                            getattr(self.config, "save_token_interval", 500_000_000)
                        )

        pbar.close()

        if local_micro_steps == 0:
            return 0.0

        return total_loss / local_micro_steps

    def train(self, resume_from_checkpoint: str = None):
        """
        The main entry point for the training loop.
        Handles epoch iteration, resuming from checkpoints, and shutdowns.

        Architecture is preserved:
            train() calls train_epoch()
        """

        if not hasattr(self, "target_tokens_seen"):
            self.target_tokens_seen = 0

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

            self.logger.info(
                f"Resuming training from epoch {self.current_epoch}, "
                f"step {self.global_step}, "
                f"tokens_seen {self.tokens_seen:,}, "
                f"target_tokens_seen {self.target_tokens_seen:,}"
            )
        else:
            self.logger.info("Starting training from scratch.")

        max_train_tokens = int(getattr(self.config, "max_train_tokens", 1_000_000_000))

        log_token_interval = int(getattr(self.config, "log_token_interval", 10_000_000))
        eval_token_interval = int(getattr(self.config, "eval_token_interval", 250_000_000))
        save_token_interval = int(getattr(self.config, "save_token_interval", 500_000_000))

        def next_boundary(current_tokens: int, interval: int) -> int:
            return ((current_tokens // interval) + 1) * interval

        self.next_log_tokens = next_boundary(self.tokens_seen, log_token_interval)
        self.next_eval_tokens = next_boundary(self.tokens_seen, eval_token_interval)
        self.next_save_tokens = next_boundary(self.tokens_seen, save_token_interval)

        self.logger.info(f"Total Epochs: {getattr(self.config, 'epochs', 5)}")
        self.logger.info(f"Max Train Tokens: {max_train_tokens:,}")
        self.logger.info(f"Gradient Accumulation Steps: {self.config.gradient_accumulation_steps}")
        self.logger.info(f"Log Token Interval: {log_token_interval:,}")
        self.logger.info(f"Eval Token Interval: {eval_token_interval:,}")
        self.logger.info(f"Save Token Interval: {save_token_interval:,}")

        try:
            for epoch in range(self.current_epoch, getattr(self.config, "epochs", 5)):
                if self.tokens_seen >= max_train_tokens:
                    self.logger.info(
                        f"Stopping before epoch {epoch + 1}: "
                        f"token budget reached "
                        f"({self.tokens_seen:,}/{max_train_tokens:,})."
                    )
                    break

                self.current_epoch = epoch

                self.logger.info(
                    f"\n{'=' * 40}\n"
                    f"Starting Epoch {epoch + 1}/{getattr(self.config, 'epochs', 5)} | "
                    f"Tokens: {self.tokens_seen:,}/{max_train_tokens:,}\n"
                    f"{'=' * 40}"
                )

                epoch_train_loss = self.train_epoch()

                self.logger.info(
                    f"Epoch {epoch + 1} completed. "
                    f"Average Training Loss: {epoch_train_loss:.4f} | "
                    f"Tokens: {self.tokens_seen:,}/{max_train_tokens:,}"
                )

                # End-of-epoch evaluation is kept, but only if budget not already exceeded.
                if self.eval_dataloader is not None and self.tokens_seen < max_train_tokens:
                    self.logger.eval("End-of-Epoch Evaluation")

                    (
                        epoch_val_loss,
                        epoch_val_ppl,
                        epoch_avg_acc_long,
                        epoch_avg_acc_short,
                        epoch_acc_diff,
                        epoch_consistency,
                        epoch_avg_depth,
                    ) = self.evaluate()

                    self.logger.eval(
                        f"Epoch {epoch + 1} | "
                        f"Validation Loss: {epoch_val_loss:.4f} | "
                        f"MaskedPerplexity: {epoch_val_ppl:.4f} | "
                        f"AccLong: {epoch_avg_acc_long:.4f} | "
                        f"AccShort: {epoch_avg_acc_short:.4f} | "
                        f"AccDiff: {epoch_acc_diff:.4f} | "
                        f"Consistency: {epoch_consistency:.4f} | "
                        f"Depth: {epoch_avg_depth:.4f}"
                    )

                    self._log_metrics({
                        "Eval/Loss": epoch_val_loss,
                        "Eval/MaskedPerplexity": epoch_val_ppl,
                        "Eval/AccLong": epoch_avg_acc_long,
                        "Eval/AccShort": epoch_avg_acc_short,
                        "Eval/AccDiff": epoch_acc_diff,
                        "Eval/Consistency": epoch_consistency,
                        "Eval/Depth": epoch_avg_depth,
                        "Eval/TokensSeen": self.tokens_seen,
                        "Eval/TargetTokensSeen": self.target_tokens_seen,
                    }, step=self.global_step)

                    if epoch_val_loss < self.best_val_loss:
                        self.best_val_loss = epoch_val_loss
                        self.save_checkpoint(is_best=True)
                        self.logger.eval(
                            f"New best validation loss: {self.best_val_loss:.4f}. "
                            f"Saved best model."
                        )

                self.save_checkpoint()

                if not hasattr(self, "training_history"):
                    self.training_history = []

                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": epoch_train_loss,
                    "global_step": self.global_step,
                    "tokens_seen": self.tokens_seen,
                    "target_tokens_seen": self.target_tokens_seen,
                })

            self.logger.info("Training completed successfully!")
            self.save_checkpoint(is_final=True)

            if self.ema is not None:
                self.logger.info("Applying EMA shadow weights to model for final artifact.")
                self.ema.copy_to_model()
                self.save_checkpoint(is_final=True)

            if self.writer is not None:
                self.writer.close()

        except KeyboardInterrupt:
            self.logger.warning("\nTraining interrupted by user!")
            self.logger.info("Saving emergency checkpoint before exiting...")

            self.save_checkpoint()

            self.logger.info("Emergency checkpoint saved. Exiting gracefully.")

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates the model on the validation dataset using the Mask-Predict strategy.
        Temporarily applies EMA shadow weights if available for a smoother, 
        more generalized evaluation metric.
        """

        self.model.eval()

        using_ema = hasattr(self, 'ema') and self.ema is not None
        if using_ema:
            self.ema.apply_shadow()

        total_val_loss = 0.0
        total_lm_loss_long = 0.0
        total_acc_long = 0.0
        total_acc_short = 0.0
        total_consistency = 0.0
        total_depth = 0.0

        pbar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            dynamic_ncols=True,
            total=self.config.eval_steps,
            leave=False
        )

        max_recursions = self.config.max_recursions
        long_steps = [1.0 / max_recursions] * max_recursions

        short_len = random.randint(1, max_recursions - 1) if max_recursions > 1 else 1
        short_steps = [1.0 / short_len] * short_len

        try:
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= self.config.eval_steps:
                    break
                    
                batch = self._move_to_device(batch)

                masked_input_ids = batch["masked_input_ids"]
                labels = batch["labels"]

                # Forward 
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    
                    # 1. Masked prediction eval
                    logits_long, loss_long, lm_loss_long, _, _, depth_long, y_long = self.model(
                        masked_input_ids=masked_input_ids,
                        labels=labels,
                        long_steps=[1.0 / self.config.max_recursions] * self.config.max_recursions,
                        short_steps=None
                    )

                    # 2. Short trajectory eval 
                    logits_short, loss_short, lm_loss_short, _, _, depth_short, y_short = self.model(
                        masked_input_ids=masked_input_ids,
                        labels=labels,
                        long_steps=short_steps,  
                        short_steps=None
                    )

                # Masked accuracy
                mask = (labels != -100)
                if mask.any():
                    acc_long = (logits_long.argmax(dim=-1)[mask] == labels[mask]).float().mean().item()
                    acc_short = (logits_short.argmax(dim=-1)[mask] == labels[mask]).float().mean().item()
                else:
                    acc_long = 0.0
                    acc_short = 0.0

                # Consistency MSE
                consistency = F.mse_loss(y_long, y_short).item()

                total_val_loss += loss_long.item()
                total_lm_loss_long += lm_loss_long.item()
                total_acc_long += acc_long
                total_acc_short += acc_short
                total_consistency += consistency
                total_depth += depth_long

                pbar.set_postfix({
                    'val_loss': f"{loss_long.item():.4f}",
                    'acc_long': f"{acc_long:.3f}",
                    'acc_short': f"{acc_short:.3f}"
                })

            num_batches = batch_idx + 1 

            avg_val_loss = total_val_loss / num_batches
            avg_lm_loss = total_lm_loss_long / num_batches
            avg_acc_long = total_acc_long / num_batches
            avg_acc_short = total_acc_short / num_batches
            avg_acc_diff = avg_acc_long - avg_acc_short
            avg_consistency = total_consistency / num_batches
            avg_depth = total_depth / num_batches
            val_perplexity = torch.exp(torch.tensor(avg_lm_loss)).item()

        finally:
            self.ema.restore()

        return (
            avg_val_loss, 
            val_perplexity,
            avg_acc_long, 
            avg_acc_short, 
            avg_acc_diff,
            avg_consistency, 
            avg_depth
        )
    
    def export_for_hf(self, export_dir, checkpoint_path, tokenizer):

        os.makedirs(export_dir, exist_ok=True)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        hf_config = TRLMConfigHF.from_internal(self.config)
        hf_config.save_pretrained(export_dir)

        save_model(
            self.model,
            os.path.join(export_dir, "model.safetensors")
        )

        tokenizer.save_pretrained(export_dir)

        self.logger.info("HF export completed!")

    def push_to_hf(self, export_dir: str, repo_id: str):
        self.logger.info(f"Pushing to Hugging Face repo: {repo_id}")

        api = HfApi()
        api.upload_folder(
            repo_id=repo_id,
            folder_path=export_dir,
            repo_type="model"
        )

        self.logger.info("Upload complete!")

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        if is_best:
            checkpoint_name = "checkpoint_best.pt"
        elif is_final:
            checkpoint_name = "checkpoint_final.pt"
        else:
            checkpoint_name = f"checkpoint_step_{self.global_step}.pt"

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),

            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "tokens_seen": self.tokens_seen,
            "target_tokens_seen": getattr(self, "target_tokens_seen", 0),

            "dataloader_state_dict": (
                self.train_dataloader.state_dict()
                if hasattr(self.train_dataloader, "state_dict")
                else None
            ),

            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,

            "config": asdict(self.config),
            "training_history": self.training_history,
        }

        if hasattr(self.model, "config"):
            checkpoint["model_config"] = asdict(self.model.config)

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", [])
        self.tokens_seen = checkpoint.get("tokens_seen", 0)
        self.target_tokens_seen = checkpoint.get("target_tokens_seen", 0)

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])

        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

        dataloader_state = checkpoint.get("dataloader_state_dict")
        if dataloader_state is not None and hasattr(self.train_dataloader, "load_state_dict"):
            self.train_dataloader.load_state_dict(dataloader_state)

        self.logger.info(f"Loaded checkpoint from step {self.global_step}")