import os
import sys
from itertools import islice
from typing import Optional, Dict, Literal, Any, List, Tuple
from tqdm import tqdm
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tqdm.auto import tqdm
from IPython.display import display, HTML

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import get_cosine_schedule_with_warmup

from configs.config import TRLMConfig
from models.trlm import TRLM
from training.optimizer import MuonAdamW

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


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
        self.warmup_steps = self.config.warmup_steps
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        
        self.device = next(self.model.parameters()).device

        self.optimizer = self._setup_optimizer()
        
        self.scheduler = self._setup_scheduler()

        self.logger = self._setup_logger()
        self.writer = self._setup_tensorboard()

        ema_decay = getattr(config, "ema_decay", 0.9999)
        self.ema = EMAModel(self.model, decay=ema_decay, device=self.device)

        self.global_step = 0
        self.current_epoch = 0
        self.tokens_seen = 0
        self.best_val_loss = float('inf')
        self.training_history: List[Dict[str, float]] = []

        self.saved_checkpoints: List[str] = []

    def _setup_scheduler(self):
        total_training_steps = self.config.max_steps_per_epoch * self.epochs
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_training_steps
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

        muon_params = []
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
                muon_params.append(param)
                
            # AdamW with weight decay for everything else (Embeddings, lm_head, q_head)
            else:
                adamw_decay_params.append(param)

        param_groups = []
        
        if muon_params:
            param_groups.append({
                "params": muon_params,
                "kind": "muon",  
                "lr": muon_lr,
                "momentum": muon_momentum,
                "beta2": muon_beta2,
                "ns_steps": muon_ns_steps,
                "weight_decay": weight_decay
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

        return MuonAdamW(param_groups)

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger."""
        logger = logging.getLogger("TRLMTrainer")
        logger.setLevel(logging.INFO)

        logger.propagate = False

        if logger.handlers:
            logger.handlers.clear()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(handler)

        return logger

    def _setup_tensorboard(self) -> Optional[Any]:
        """Setup TensorBoard writer."""
        if not self.config.use_tensorboard or not TENSORBOARD_AVAILABLE:
            return None
        
        os.makedirs(self.config.log_dir, exist_ok=True)

        return SummaryWriter(log_dir=self.config.log_dir, flush_secs=5)

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all configured destinations."""

        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)

    def train_step(self, batch: Dict[str, torch.Tensor], step_idx: int):
        """
        Executes a single training iteration: forward pass, backward pass, and 
        conditional optimizer step based on gradient accumulation.
        """

        batch = self._move_to_device(batch)
        masked_input_ids = batch["masked_input_ids"]
        labels = batch["labels"]

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits, loss, lm_loss, q_loss, recursion_depth = self.model(
                masked_input_ids=masked_input_ids,
                labels=labels
            )

            scaled_loss = loss / self.config.gradient_accumulation_steps

        scaled_loss.backward()

        # Optimizer Step
        grad_norm = 0.0
        if (step_idx + 1) % self.config.gradient_accumulation_steps == 0:
            
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

        safe_lm_loss = torch.clamp(lm_loss, max=20.0) 
        perplexity = torch.exp(safe_lm_loss).item()

        return {
            "loss": loss.item(),
            "lm_loss": lm_loss.item(),
            "q_loss": q_loss.item(),
            "perplexity": perplexity,
            "recursion_depth": recursion_depth,
            "grad_norm": grad_norm,
            "batch_tokens": masked_input_ids.numel() # Total tokens processed
        }

    def train_epoch(self):
        """
        Runs one complete pass over the training dataset.
        Handles progress tracking, logging, periodic evaluation, and checkpointing.
        """
        
        self.model.train()
        
        total_loss = 0.0

        pbar = tqdm(
            total=self.config.max_steps_per_epoch,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            dynamic_ncols=True
        )

        for step_idx, batch in enumerate(
                islice(self.train_dataloader, self.config.max_steps_per_epoch)
        ):
            
            step_metrics = self.train_step(batch, step_idx)
            
            total_loss += step_metrics["loss"]
            self.tokens_seen += step_metrics["batch_tokens"]
            ppl = step_metrics["perplexity"]

            avg_loss = total_loss / (step_idx + 1)

            pbar.update(1)

            pbar.set_postfix({
                'step': self.global_step,
                'loss': f"{avg_loss:.4f}",
                'lm_loss': f"{step_metrics['lm_loss']:.4e}",
                'q_loss': f"{step_metrics['q_loss']:.4e}",
                'ppl': f"{ppl:.4f}",
                'grad_norm': f"{step_metrics['grad_norm']:.4e}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.4e}",
                'depth': f"{step_metrics['recursion_depth']}"
            })

            is_opt_step = (step_idx + 1) % self.config.gradient_accumulation_steps == 0

            if is_opt_step:
                
                if self.global_step % getattr(self.config, 'logging_steps', 10) == 0:
                    
                    log_dict = {
                        "Train/Loss": avg_loss,
                        "Train/Lm_loss": f"{step_metrics['lm_loss']:.4e}",
                        "Train/Q_loss": f"{step_metrics['q_loss']:.4e}",
                        "Train/Perplexity": ppl,
                        "Train/LearningRate": self.optimizer.param_groups[0]['lr'],
                        "Metrics/RecursionDepth": step_metrics["recursion_depth"],
                        "Metrics/TokensSeen": self.tokens_seen
                    }

                    log_dict["Metrics/GradNorm"] = step_metrics["grad_norm"]

                    self._log_metrics(log_dict)

                    self.logger.info(
                        f"Step {self.global_step} | Loss: {avg_loss:.4f} | "
                        f"Lm_loss: {step_metrics['lm_loss']:.4e} | "
                        f"Q_loss: {step_metrics['q_loss']:.4e} | "
                        f"Ppl: {ppl} | "
                        f"Grad norm: {step_metrics['grad_norm']} | "
                        f"Depth: {step_metrics['recursion_depth']} | "
                        f"Tokens: {self.tokens_seen:.2e}"
                    )
                
                # Evaluation
                if self.global_step % getattr(self.config, 'eval_interval', 10) == 0:
                    val_loss, epoch_val_ppl = self.evaluate()

                    self.logger.info(f"Step {self.global_step} Validation Loss: {val_loss:.4f} | Perplexity: {epoch_val_ppl:.2f}")

                    self._log_metrics({
                        "Eval/Loss": val_loss,
                        "Eval/Perplexity": epoch_val_ppl
                    })
                    
                    self.model.train() 
                    
                    # Check if this is the best model so far
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                
                # Checkpointing
                if self.global_step % getattr(self.config, 'save_steps', 1000) == 0:
                    self.save_checkpoint()

        pbar.close()
        
        return total_loss / self.config.max_steps_per_epoch

    def train(self, resume_from_checkpoint: str = None):
        """
        The main entry point for the training loop.
        Handles epoch iteration, resuming from checkpoints, and shutdowns.
        """

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            self.logger.info(f"Resuming training from epoch {self.current_epoch}, step {self.global_step}")
        else:
            self.logger.info("Starting training from scratch.")

        self.logger.info(f"Total Epochs: {getattr(self.config, 'epochs', 5)}")
        self.logger.info(f"Gradient Accumulation Steps: {self.config.gradient_accumulation_steps}")

        try:
            for epoch in range(self.current_epoch, getattr(self.config, 'epochs', 5)):
                self.current_epoch = epoch
                
                self.logger.info(f"\n{'='*40}\nStarting Epoch {epoch+1}/{getattr(self.config, 'epochs', 5)}\n{'='*40}")

                # execute epoch
                epoch_train_loss = self.train_epoch()

                self.logger.info(f"Epoch {epoch+1} completed. Average Training Loss: {epoch_train_loss:.4f}")

                # End-of-Epoch Evaluation
                epoch_val_loss, epoch_val_ppl = self.evaluate()
    
                self.logger.info(f"Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f} | Perplexity: {epoch_val_ppl:.2f}")

                self._log_metrics({
                    "Eval/Loss": epoch_val_loss,
                    "Eval/Perplexity": epoch_val_ppl
                })

                # Update best model state
                if epoch_val_loss < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}. Saved best model.")

                self.save_checkpoint()

                # Record history
                if not hasattr(self, 'training_history'):
                    self.training_history = []
                    
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'global_step': self.global_step
                })

            self.logger.info("Training completed successfully!")
            self.save_checkpoint(is_final=True)

            # EMA Finalization
            if self.ema is not None:
                self.logger.info("Applying EMA shadow weights to model for final artifact.")
                self.ema.copy_to_model()
                self.save_checkpoint(is_final=True) # Overwrite final checkpoint with smoothed weights

            if self.writer is not None:
                self.writer.close() 

        except KeyboardInterrupt:
            self.logger.warning("\nTraining interrupted by user (KeyboardInterrupt)!")
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

        pbar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            dynamic_ncols=True,
            total=self.config.eval_steps,
            leave=False
        )

        try:
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if batch_idx >= self.config.eval_steps:
                    break
                
                batch = self._move_to_device(batch)

                pbar.update(1)

                masked_input_ids = batch["masked_input_ids"]
                labels = batch["labels"]

                # Forward 
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    logits, loss, lm_loss, q_loss, recursion_depth = self.model(
                        masked_input_ids=masked_input_ids,
                        labels=labels
                    )

                total_val_loss += loss.item()
                total_val_lm_loss += lm_loss.item()
                
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

            avg_val_loss = total_val_loss / self.config.eval_steps
            avg_val_lm_loss = total_val_lm_loss / self.config.eval_steps
            
            val_perplexity = torch.exp(torch.tensor(avg_val_lm_loss)).item()

        finally:
            self.ema.restore()

        return avg_val_loss, val_perplexity

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        
        # Determine checkpoint name
        if is_best:
            checkpoint_name = "checkpoint_best.pt"
        elif is_final:
            checkpoint_name = "checkpoint_final.pt"
        else:
            checkpoint_name = f"checkpoint_step_{self.global_step}.pt"

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name

        # Save checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'training_history': self.training_history,
        }

        if hasattr(self.model, 'config'):
            checkpoint['model_config'] = asdict(self.model.config)

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Track saved checkpoints (for cleanup)
        if not is_best and not is_final:
            self.saved_checkpoints.append(str(checkpoint_path))

            # Remove old checkpoints if over limit
            if (self.config.save_total_limit > 0 and
                    len(self.saved_checkpoints) > self.config.save_total_limit):
                old_checkpoint = self.saved_checkpoints.pop(0)
                if Path(old_checkpoint).exists():
                    Path(old_checkpoint).unlink()
                    self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])

        self.logger.info(f"Loaded checkpoint from step {self.global_step}")