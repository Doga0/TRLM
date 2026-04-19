import os
from typing import Optional, Dict, Literal, Any, List, Tuple
from tqdm import tqdm
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

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
                 epochs: int,
                 warmup_steps: int,
                 gradient_accumulation_steps: int,
                 eval_dataloader = None
            ):
        
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizer = self._setup_optimizer()
        
        self.scheduler = self._setup_scheduler()

        self.logger = self._setup_logger()
        self.writer = self._setup_tensorboard()

        self.global_step = 0
        self.current_epoch = 0
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

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

        return logger

    def _setup_tensorboard(self) -> Optional[Any]:
        """Setup TensorBoard writer."""
        if not self.config.use_tensorboard or not TENSORBOARD_AVAILABLE:
            return None

        return SummaryWriter(log_dir=self.config.log_dir)

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all configured destinations."""
        self.logger.info(f"Step {self.global_step}: {metrics}")

        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)

    def train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch = self._move_to_device(batch)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss, metrics = self.model(
                input_ids=batch['input_ids'],
                response_mask=batch['response_mask'],
                position_ids=batch.get('position_ids'),
                mask=batch.get('attention_mask')
            )

            # loss, metrics = self.compute_loss(all_logits, all_q_logits, batch, all_masks)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        loss.backward()

        metrics['loss'] = loss.item() * self.gradient_accumulation_steps

        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            
            # Gradient clipping
            max_grad_norm = getattr(self.config, "max_grad_norm", 1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_grad_norm
            )
            metrics['grad_norm'] = grad_norm.item()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1

            metrics['lr'] = self.scheduler.get_last_lr()[0]

        return metrics

    def train_epoch(self):
        self.model.train()
        
        epoch_metrics = {}
        total_loss = 0.0
        steps_in_this_epoch = 0

        pbar = tqdm(
            total=self.config.max_steps_per_epoch,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            dynamic_ncols=True
        )

        for batch_idx, batch in enumerate(self.train_dataloader):
            
            step_metrics = self.train_step(batch, batch_idx)

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                steps_in_this_epoch += 1

                pbar.update(1)

                total_loss += step_metrics['loss']
                for k, v in step_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                
                pbar.set_postfix({
                    'train_loss': f"{step_metrics['loss']:.3f}",
                    'mdm': f"{step_metrics.get('loss/mdm_diffusion', 0):.3f}",
                    'ponder_time': f"{step_metrics.get('metrics/avg_ponder_time', 0):.1f}",
                    'lr': f"{step_metrics.get('lr', 0):.2e}",
                    'grad_norm': f"{step_metrics.get('grad_norm', 0):.1f}",
                    'q_halting_bce': f"{step_metrics.get('loss/q_halting_bce'):.2e}",
                    'halt_ratio': f"{step_metrics.get('metrics/halt_ratio'):.2e}",
                    'final_step_acc': f"{step_metrics.get('metrics/final_step_acc'):.2e}"
                })

                # log training metrics
                log_interval = getattr(self.config, "log_interval", 50)
                if self.global_step % log_interval == 0:
                    logs = {}
                    for k, v in step_metrics.items():
                        if '/' in k:
                            category, name = k.split('/', 1)
                            logs[f"train_{category}/{name}"] = v
                        else:
                            logs[f"train/{k}"] = v
                            
                    logs['train/global_step'] = self.global_step
                    self._log_metrics(logs)

                # run mid epoch evaluation
                eval_interval = getattr(self.config, "eval_interval", 500)
                if self.global_step % eval_interval == 0 and self.eval_dataloader is not None:
                    val_metrics = self.evaluate(skip_generation=True)
                    
                    val_logs = {}
                    for k, v in val_metrics.items():
                        if '/' in k:
                            category, name = k.split('/', 1)
                            val_logs[f"val_{category}/{name}"] = v
                        else:
                            val_logs[f"val/{k}"] = v
                            
                    val_logs['val/global_step'] = self.global_step
                    self._log_metrics(val_logs)

                # Stop the epoch when we reach the step limit
                if steps_in_this_epoch >= self.config.max_steps_per_epoch:
                    break
                
        pbar.close()

        denom = max(1, len(self.train_dataloader))
        avg_epoch_metrics = {k: v / denom for k, v in epoch_metrics.items()}
        avg_epoch_metrics['train/epoch_loss'] = total_loss / denom
        
        return avg_epoch_metrics

    def train(self):
        self.logger.info(f"Starting TRLM Training")
        self.logger.info(f"Total Epochs: {self.epochs}")
        self.logger.info(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")

        self.device = next(self.model.parameters()).device

        best_metrics = {}
        patience_counter = 0
        early_stopping_patience = getattr(self.config, "early_stopping_patience", -1)

        # Ensure gradients are zeroed before starting
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train for one epoch
            epoch_metrics = self.train_epoch()
            self.training_history.append(epoch_metrics)

            self.logger.info(f"Epoch {epoch + 1} Training Loss: {epoch_metrics.get('train/epoch_loss', 0):.4f}")

            # Evaluate
            if self.eval_dataloader is not None:
                val_metrics = self.evaluate()

                for k, v in val_metrics.items():
                    self._log_metrics({f"val/{k}": v})

                self.logger.info(f"Epoch {epoch + 1} Validation Loss: {val_metrics['loss']:.4f}")
            
                # Checkpointing & Early Stopping Logic
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    best_metrics = val_metrics.copy()
                    
                    self.save_checkpoint(is_best=True)
                    patience_counter = 0
                    self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}. Checkpoint saved.")
                else:
                    patience_counter += 1

                if 0 < early_stopping_patience <= patience_counter:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                    break

            else:
                save_every_epoch = getattr(self.config, "save_every_epoch", True)
                if save_every_epoch:
                    self.save_checkpoint(is_best=False)

        self.save_checkpoint(is_final=True)
        self.logger.info("Training complete.")

        if self.writer is not None:
            self.writer.close()

        return best_metrics

    @torch.no_grad()
    def evaluate(self, skip_generation: bool = False):
    
        self.model.eval()
        val_metrics = {}
        total_loss = 0.0
        total_exact_match = 0.0
        num_batches = 0
        
        # We might only want to run heavy generation (EM) on a subset to save time
        do_generation = getattr(self.config, "eval_generation", True) and not skip_generation
        
        pbar = tqdm(self.eval_dataloader, desc="Evaluating", dynamic_ncols=True)
        
        for batch in pbar:
            batch = self._move_to_device(batch)
            num_batches += 1
            
            # Forward Pass (Deep Supervision / Step-wise Loss)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                loss, step_metrics = self.model(
                    input_ids=batch['input_ids'],
                    response_mask=batch['response_mask'],
                    position_ids=batch.get('position_ids'),
                    mask=batch.get('attention_mask')
                )
                
            total_loss += loss.item()
            
            # Aggregate step-wise metrics
            for k, v in step_metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v

            # Generative Pass (Exact Match / Final Performance)
            # This is the "Inference" mode. It starts from [MASK] and iteratively solves.
            if do_generation:
                
                prompt_len = (~batch['response_mask']).sum(dim=1).min().item()
                prompts = batch['input_ids'][:, :prompt_len]
                
                # Run the Diffusion Sampling Loop
                generated_response = self.model.generate(
                    prompt_ids=prompts,
                    max_response_len=batch['input_ids'].size(1) - prompt_len,
                    num_steps=getattr(self.config, "eval_sampling_steps", 32)
                )
                
                # Compare generated vs ground truth response
                target_response = batch['input_ids'][:, prompt_len:]
                
                # Exact Match: Every token in the response must be correct
                is_correct = (generated_response == target_response).all(dim=1)
                total_exact_match += is_correct.float().mean().item()

        avg_val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        avg_val_metrics['loss'] = total_loss / num_batches
        
        if do_generation:
            avg_val_metrics['exact_match_ratio'] = total_exact_match / num_batches
            self.logger.info(f"Validation Exact Match (EM): {avg_val_metrics['exact_match_ratio']:.2%}")

        self.model.train()
        return avg_val_metrics

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        
        # Determine checkpoint name
        if is_best:
            checkpoint_name = "checkpoint_best.pt"
        elif is_final:
            checkpoint_name = "checkpoint_final.pt"
        else:
            checkpoint_name = f"checkpoint_step_{self.global_step}.pt"

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