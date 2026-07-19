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


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class ReasoningModule(nn.Module):
    def __init__(self, config: TRLMConfig):
        super().__init__()

        # Projects the concatenated x, y, and z states back to hidden_dim.
        self.combiner = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.use_trajectory_conditioning = getattr(config, "use_trajectory_conditioning", True)

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
                attn_dropout=config.attn_dropout,
                transformer_dropout=config.transformer_dropout,
                use_adaln=self.use_trajectory_conditioning
            ) for _ in range(config.num_layers)
        ])

        self.step_dropout = nn.Dropout(config.step_dropout)

        self.max_recursions = getattr(config, 'max_recursions', 8)

        # self.gate = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.time_embedder = TimestepEmbedder(config.hidden_dim)
        self.dt_embedder = TimestepEmbedder(config.hidden_dim)

        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=True),
        )

        # with torch.no_grad():
        #     self.gate.weight.zero_()
        #     self.gate.bias.fill_(-2.0) 

    def init_adaln(self, hidden_dim):
        with torch.no_grad():
            self.cond_mlp[1].weight.zero_()
            self.cond_mlp[1].bias.zero_()
            self.cond_mlp[1].bias[: 2 * hidden_dim].fill_(0.1)

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        z: torch.Tensor = None,
        steps: Optional[list[float]] = None,
    ) -> torch.Tensor:
        
        batch_size, seq_len, hidden_dim = x.shape

        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        # Initialize z state to zeros if this is the very first step
        if z is None:
            z = torch.zeros_like(x, device=x.device, dtype=x.dtype)

        # If 'y' is a single pooled state (Batch, Dim), broadcast it to sequence length
        if y.dim() == 2:
            y = y.unsqueeze(1).expand(-1, seq_len, -1)

        if steps is None:
            steps = [1.0 / self.max_recursions] * self.max_recursions

        # normalized trajectory time
        ti = torch.zeros(batch_size, device=x.device, dtype=torch.float32)

        # Latent Recursion Loop: refine z 'step' times
        for dt in steps:
            cond = None
            if self.use_trajectory_conditioning:
                dt_base = torch.ones_like(ti) * dt
                te = self.time_embedder(ti)
                dte = self.dt_embedder(dt_base)
                c = te + dte

                # calculate adaln conditioning
                cond = self.cond_mlp(c)  # [B, 4*H]

            # Combine x, y, z
            combined = torch.cat([x, y, z], dim=-1)
            hidden_state = self.combiner(combined)

            for layer in self.layers:
                hidden_state = layer(
                    hidden_state, 
                    position_ids=position_ids,
                    cond=cond  
                )

            hidden_state = self.step_dropout(hidden_state)

            # z = z + gate * hidden_state
            z = z + hidden_state
            ti = ti + dt

        return z

class TRLM(nn.Module):
    def __init__(self, config: TRLMConfig):
        super().__init__()

        self.config = config
        self.mask_token_id = config.mask_token_id

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Tie

        self.short_lm_weight = getattr(config, "short_lm_weight", 0.1)
        self.consistency_weight = getattr(config, "consistency_weight", 0.1)
        self.max_recursions = getattr(config, "max_recursions", 8)
        self.use_trajectory_conditioning = getattr(config, "use_trajectory_conditioning", True)

        self.reasoning_module = ReasoningModule(config)

        self.answer_combiner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.apply(self._init_weights)

        self.reasoning_module.init_adaln(config.hidden_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _initialize_states(self, input_ids):
        """Initializes the x (input), y (answer), and z (reasoning) states."""
        x = self.embed_tokens(input_ids)
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)
        return x, y, z
    
    def _update_answer_state(self, y, z, cond=None):
        """Updates the answer state 'y' using the reasoning state 'z'."""
        combined = torch.cat([y, z], dim=-1)           # [B, S, H*2]
        h = self.answer_combiner(combined)             # [B, S, H]

        seq_len = h.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=h.device).unsqueeze(0)
        
        # Pass through the shared transformer blocks to leverage shared weights
        for layer in self.reasoning_module.layers:
            h = layer(h, position_ids=position_ids, cond=cond)
        
        return y + h  # Residual connection for stability
    
    def _compute_step_loss(
            self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
        ):

        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100 
        )
        
        return lm_loss

    def forward(
            self, 
            masked_input_ids, 
            labels=None,
            long_steps: Optional[list[float]] = None,
            short_steps: Optional[list[float]] = None
        ):
        x, y, z = self._initialize_states(masked_input_ids)

        if long_steps is None:
            long_steps = [1.0 / self.max_recursions] * self.max_recursions

        z_long = self.reasoning_module(x, y, z, steps=long_steps)
        y_long = self._update_answer_state(y, z_long)
        y_long_norm = self.final_norm(y_long)
        logits_long = self.lm_head(y_long_norm)

        total_loss = None
        lm_loss_long = None
        lm_loss_short = torch.tensor(0.0, device=masked_input_ids.device)
        consistency_loss = torch.tensor(0.0, device=masked_input_ids.device)

        if labels is not None:
            lm_loss_long = self._compute_step_loss(logits_long, labels)
            total_loss = lm_loss_long

            # Short trajectory + consistency
            if short_steps is not None:
                z_short = self.reasoning_module(x, y, z, steps=short_steps)
                y_short = self._update_answer_state(y, z_short)
                y_short_norm = self.final_norm(y_short)
                logits_short = self.lm_head(y_short_norm)

                lm_loss_short = self._compute_step_loss(logits_short, labels)

                # Shortcut-consistency: short path should match long path
                consistency_loss = F.mse_loss(y_short_norm, y_long_norm.detach())

                total_loss = (
                    lm_loss_long
                    + self.short_lm_weight * lm_loss_short
                    + self.consistency_weight * consistency_loss
                )

        return (
            logits_long,
            total_loss,
            lm_loss_long,
            lm_loss_short,
            consistency_loss,
            len(long_steps),
            y_long_norm,
        )
        
    @torch.no_grad()
    def _filter_generation_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_k: int | None = None,
        forbid_token_ids: list[int] | None = None,
    ):
        """
        logits: [B, S, V]
        Returns filtered logits.
        """
        logits = logits.float()

        if forbid_token_ids is not None:
            vocab_size = logits.size(-1)
            for token_id in forbid_token_ids:
                if token_id is not None and 0 <= int(token_id) < vocab_size:
                    logits[..., int(token_id)] = -float("inf")

        if temperature is not None and temperature > 0:
            logits = logits / temperature

        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            values, indices = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(-1, indices, values)
            logits = filtered

        return logits


    @torch.no_grad()
    def generate(
        self,
        masked_input_ids: torch.Tensor,
        recursion_steps: int | None = None,
        forbid_mask_token: bool = True,
        forbid_token_ids: list[int] | None = None,
        return_confidence: bool = False,
    ):
        self.eval()

        if recursion_steps is None:
            recursion_steps = self.max_recursions

        recursion_steps = max(1, int(recursion_steps))
        steps = [1.0 / recursion_steps] * recursion_steps

        logits, *_ = self.forward(
            masked_input_ids=masked_input_ids,
            labels=None,
            long_steps=steps,
            short_steps=None,
        )

        forbidden = list(forbid_token_ids or [])

        if forbid_mask_token:
            forbidden.append(self.mask_token_id)

        logits = self._filter_generation_logits(
            logits=logits,
            temperature=0.0,
            top_k=None,
            forbid_token_ids=forbidden,
        )

        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, predicted_ids = probabilities.max(dim=-1)

        output_ids = masked_input_ids.clone()
        answer_mask = masked_input_ids.eq(self.mask_token_id)

        output_ids[answer_mask] = predicted_ids[answer_mask]

        if return_confidence:
            output_confidence = torch.ones_like(
                masked_input_ids,
                dtype=torch.float32,
            )
            output_confidence[answer_mask] = confidence[answer_mask]
            return output_ids, output_confidence

        return output_ids
        
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
            "model_state_dict": self.state_dict(),
            'ema_state_dict': self.ema.state_dict()
        }, os.path.join(path, "model.pt"))
