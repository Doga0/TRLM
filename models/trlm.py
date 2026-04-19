import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from dataclasses import asdict
from typing import Optional, List, Dict, Tuple

from models.transformer_blocks import TransformerBlock
from models.layers import CastedLinear, RMSNorm
from models.utils import trunc_normal_init_

from configs.config import TRLMConfig


class QHead(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        self.norm = RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize output to predict 0 initially
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        
        z = self.norm(z)
        # Global average pooling over sequence
        z_pooled = z.mean(dim=1)
        # Predict halting probability
        q_hat = self.mlp(z_pooled).squeeze(-1)
        return q_hat

class ReasoningModule(nn.Module):
    def __init__(self, config: TRLMConfig):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                rms_norm_eps=config.rms_norm_eps,
                attn_type=config.attn,
                num_heads=config.num_heads,
                max_seq_len=config.max_seq_len,
                num_groups=config.num_groups,
                swiglu_intermediate_size=config.swiglu_intermediate_size,
                swiglu_hidden_act=config.swiglu_hidden_act,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])

    def forward(
        self, 
        state: torch.Tensor, 
        injection: torch.Tensor, 
        position_ids: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        is_causal: bool = False
    ) -> torch.Tensor:
        
        # TODO: try learned gating mechanism
        x = 0.5 * state + 0.5 * injection
        
        for layer in self.layers:
            x = layer(x, position_ids=position_ids, mask=mask, is_causal=is_causal)
            
        return x

class TRLM(nn.Module):
    def __init__(self, config: TRLMConfig):
        super().__init__()

        self.config = config
        self.mask_token_id = config.mask_token_id

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.q_head = QHead(embed_dim=config.hidden_dim)

        self.reasoning_module = ReasoningModule(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _latent_recursion(
        self, 
        x_emb: torch.Tensor, 
        z_H: torch.Tensor, 
        z_L: torch.Tensor, 
        position_ids: torch.Tensor, 
        mask: Optional[torch.Tensor], 
        is_causal: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Recursive Latent Reasoning: Update z_L n times
        # z_L <- f(z_L + (x_emb + z_H))
        for _ in range(self.config.n_cycles):
            z_L = self.reasoning_module(
                state=z_L, 
                injection=x_emb + z_H, 
                position_ids=position_ids, 
                mask=mask, 
                is_causal=is_causal
            )
            
        # Refine Output Answer: Update z_H 1 time
        # z_H <- f(z_H + z_L)
        z_H = self.reasoning_module(
            state=z_H, 
            injection=z_L, 
            position_ids=position_ids, 
            mask=mask, 
            is_causal=is_causal
        )
        
        return z_H, z_L
    
    def _deep_recursion(
        self, 
        x_emb: torch.Tensor, 
        z_H: torch.Tensor, 
        z_L: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # T-1 loops without gradients to improve y (z_H) and z (z_L) efficiently
        with torch.no_grad():
            for _ in range(self.config.t_cycles - 1):
                z_H, z_L = self._latent_recursion(x_emb, z_H, z_L, position_ids, mask, is_causal)
        
        # 1 loop with gradients for backpropagation
        z_H, z_L = self._latent_recursion(x_emb, z_H, z_L, position_ids, mask, is_causal)
        
        # Detach states for the next supervision step to avoid OOM
        z_H_detached = z_H.detach()
        z_L_detached = z_L.detach()
        
        return z_H_detached, z_L_detached, z_H, z_L
    
    def _low_confidence_remask(self, r_hat, logits, response_mask, target_t):
        batch_size, seq_len = r_hat.shape
        r_next = r_hat.clone()
        
        # Calculate confidences
        probs = torch.softmax(logits, dim=-1)
        confidences, _ = probs.max(dim=-1)
        
        # We only want to mask response tokens
        confidences[~response_mask] = float('inf')
        
        # Number of tokens to mask per sequence
        resp_lengths = response_mask.sum(dim=1).float()
        num_to_mask = (resp_lengths * target_t).long()
        
        for i in range(batch_size):
            k = num_to_mask[i].item()
            if k > 0:
                _, indices = torch.topk(confidences[i], k, largest=False)
                r_next[i, indices] = self.mask_token_id
                
        return r_next

    def forward(self, input_ids: torch.Tensor, response_mask: torch.Tensor, position_ids=None, mask=None):
        
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if mask is not None:
            mask = mask.to(torch.bool)
            
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)

        # Initialize fully masked response
        r_t = input_ids.clone()
        r_t[response_mask] = self.mask_token_id
        
        # State Initialization
        x_emb = self.embed_tokens(r_t)
        y = x_emb.clone()
        z = torch.zeros_like(y)
        
        total_loss = 0.0
        metrics = {}
        
        n_sup = getattr(self.config, "n_sup", 8)  # Denoising steps per batch
        
        for s in range(n_sup):
            t_current = 1.0 - (s / n_sup)
            t_next = 1.0 - ((s + 1) / n_sup)
            
            x_emb = self.embed_tokens(r_t)
            
            # Deep Recursion
            y_det, z_det, y_grad, z_grad = self._deep_recursion(x_emb, y, z, position_ids, mask, is_causal=False)
            
            logits = self.lm_head(y_grad)
            q_logits = self.q_head(y_grad)
            
            # Masked Diffusion Loss
            is_masked = (r_t == self.mask_token_id) & response_mask
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), 
                input_ids.reshape(-1), reduction='none'
            ).reshape(batch_size, seq_len)
            
            # Reweighted by 1/t
            t_weight = 1.0 / max(t_current, 1e-5)
            resp_lengths = response_mask.sum(dim=1).float().clamp(min=1)
            
            mdm_loss = (ce_loss * is_masked.float()).sum(dim=1) / resp_lengths * t_weight
            mdm_loss = mdm_loss.mean()
            
            # Halting Loss
            r_hat = logits.argmax(dim=-1)
            correct_tokens = (r_hat == input_ids) | ~response_mask
            all_correct = correct_tokens.all(dim=1).float() # 1 if decoded sequence is perfect
            halt_loss = F.binary_cross_entropy_with_logits(q_logits, all_correct)
            
            step_loss = mdm_loss + getattr(self.config, "lambda_halt", 1.0) * halt_loss
            total_loss += step_loss
            
            metrics[f'step_{s}_mdm_loss'] = mdm_loss.item()
            metrics[f'step_{s}_acc'] = correct_tokens.float().mean().item()
            
            # Early halting mechanism to save compute
            if (torch.sigmoid(q_logits) > getattr(self.config, "halt_threshold", 0.9)).all() and s >= 2:
                break
                
            # Remask for next denoising iteration
            if s < n_sup - 1:
                with torch.no_grad():
                    # Copy-through constraint: force prompt tokens to remain correct
                    r_hat[~response_mask] = input_ids[~response_mask]
                    r_t = self._low_confidence_remask(r_hat, logits, response_mask, t_next)
            
            # Detach states to prevent BPTT OOM across denoising steps
            y, z = y_det, z_det
            
        return total_loss / (s + 1), metrics
        
    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_response_len: int, num_steps: int = 32, position_ids=None):
        was_training = self.training
        self.eval()
        
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device
        
        try:
            # Initialize fully masked sequence
            masked_response = torch.full((batch_size, max_response_len), self.mask_token_id, device=device)
            r_t = torch.cat([prompt_ids, masked_response], dim=1)
            
            response_mask = torch.zeros_like(r_t, dtype=torch.bool)
            response_mask[:, prompt_ids.size(1):] = True
            
            x_emb = self.embed_tokens(r_t)
            y = x_emb.clone()
            z = torch.zeros_like(y)
            
            for s in range(num_steps):
                t_next = 1.0 - ((s + 1) / num_steps)
                x_emb = self.embed_tokens(r_t)
                
                # Deep Recursion without Gradients
                for _ in range(self.config.t_cycles):
                    y, z = self._latent_recursion(x_emb, y, z, position_ids, mask=None)
                    
                logits = self.lm_head(y)
                r_hat = logits.argmax(dim=-1)
                
                copy_mask = (r_t != self.mask_token_id) & response_mask
                r_hat[copy_mask] = r_t[copy_mask]
                r_hat[~response_mask] = prompt_ids.view(-1)
                
                if s == num_steps - 1:
                    r_t = r_hat
                    break
                    
                r_t = self._low_confidence_remask(r_hat, logits, response_mask, t_next)
                
            return r_t[:, prompt_ids.size(1):]  # Return only the decoded response
            
        finally:
            if was_training:
                self.train()
        
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: TRLMConfig) -> "TRLM":
        """Create TRM from configuration."""
        return cls(config)

    @classmethod
    def from_pretrained(cls, path: str) -> "TRLM":
        """Load pretrained TRM model."""

        checkpoint = torch.load(path, map_location="cpu")
        config = TRLMConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""

        os.makedirs(path, exist_ok=True)

        torch.save({
            "config": asdict(self.config),
            "model_state_dict": self.state_dict()
        }, os.path.join(path, "model.pt"))

if __name__ == "__main__":
    print("🚀 Starting TRLM Sanity Check...")
    torch.manual_seed(42)

    test_config = TRLMConfig(
        vocab_size=100,          
        hidden_dim=64,           
        num_heads=2,
        num_groups=2,
        max_seq_len=32,
        num_layers=2,
        swiglu_intermediate_size=128,
        n_cycles=2,              
        t_cycles=2,
        max_act_steps=4          
    )

    model = TRLM(test_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"✅ Model instantiated. Total parameters: {model.num_parameters():,}")

    # Generate Dummy Data
    batch_size = 2
    seq_len = 16
    
    input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    print(f"📊 Dummy Data created. Input shape: {input_ids.shape}")

    # ==========================================
    # TEST 1: TRAINING FORWARD PASS
    # ==========================================
    print("\n🧪 TEST 1: Forward Pass (Training Mode)")
    model.train()
    optimizer.zero_grad()
    
    all_logits, all_q_logits = model(
        input_ids=input_ids, 
        position_ids=position_ids, 
        is_causal=False 
    )
    
    expected_logits_shape = (batch_size, test_config.max_act_steps, seq_len, test_config.vocab_size)
    expected_q_shape = (batch_size, test_config.max_act_steps)
    
    assert all_logits.shape == expected_logits_shape, f"Logits shape mismatch: {all_logits.shape} != {expected_logits_shape}"
    assert all_q_logits.shape == expected_q_shape, f"Q-logits shape mismatch: {all_q_logits.shape} != {expected_q_shape}"
    print("✅ Forward pass shapes are perfectly aligned.")

    # ==========================================
    # TEST 2: DEEP SUPERVISION LOSS & BACKWARD PASS
    # ==========================================
    print("\n🧪 TEST 2: Deep Supervision Loss & Gradient Flow")
    
    total_loss = 0
    
    for step in range(test_config.max_act_steps):
        step_logits = all_logits[:, step, :, :]
        step_q_logits = all_q_logits[:, step]  
        
        # A. Language Modeling Loss (Cross Entropy)
        ce_loss = F.cross_entropy(
            step_logits.reshape(-1, test_config.vocab_size), 
            target_ids.reshape(-1)
        )
        
        # B. Halting Logic Loss (BCE)
        with torch.no_grad():
            predictions = step_logits.argmax(dim=-1)
            is_correct = (predictions == target_ids).all(dim=-1).float() 
            
        q_loss = F.binary_cross_entropy_with_logits(step_q_logits, is_correct)
        
        total_loss += (ce_loss + q_loss)
        
    print(f"ℹ️ Calculated Initial Loss: {total_loss.item():.4f}")
    
    total_loss.backward()
    optimizer.step()
    
    assert model.embed_tokens.weight.grad is not None, "❌ No gradient in embed_tokens!"
    assert model.lm_head.weight.grad is not None, "❌ No gradient in lm_head!"
    assert model.q_head.mlp[-1].weight.grad is not None, "❌ No gradient in q_head!"
    
    print("✅ Backward pass completed successfully. Gradients are flowing!")

    # ==========================================
    # TEST 3: INFERENCE (GENERATE FUNCTION)
    # ==========================================
    print("\n🧪 TEST 3: Inference Generation & Early Stopping")
    
    predicted_tokens = model.generate(
        input_ids=input_ids,
        position_ids=position_ids,
        is_causal=False
    )
    
    expected_gen_shape = (batch_size, seq_len)
    assert predicted_tokens.shape == expected_gen_shape, f"Generate shape mismatch: {predicted_tokens.shape} != {expected_gen_shape}"
    
    print(f"✅ Generation successful. Output shape: {predicted_tokens.shape}")
    print("\n🎉 ALL TESTS PASSED! Model is ready for real training.")