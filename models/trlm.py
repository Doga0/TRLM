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


@dataclass
class SentenceACTState:
    halted: torch.Tensor          # (B,)
    halt_prob_accum: torch.Tensor # (B,)
    remainders: torch.Tensor      # (B,)
    n_updates: torch.Tensor       # (B,)
    output_accum: torch.Tensor    # (B, T, C)


class ACTHalting(nn.Module):
    def __init__(self, hidden_dim: int, halt_threshold: float = 1.0 - 1e-6):
        super().__init__()
        self.halt_head = nn.Linear(hidden_dim, 1)
        self.halt_threshold = halt_threshold

    def init_state(self, B, T, C, device):
        return SentenceACTState(
            halted=torch.zeros(B, dtype=torch.bool, device=device),
            halt_prob_accum=torch.zeros(B, device=device),
            remainders=torch.zeros(B, device=device),
            n_updates=torch.zeros(B, device=device),
            output_accum=torch.zeros(B, T, C, device=device),
        )

    def step(
        self,
        x: torch.Tensor,  # (B, T, C)
        state: SentenceACTState
    ) -> Tuple[SentenceACTState, bool]:

        # sentence representation
        h = x[:, 0, :]  # (B, C) TODO: use attention pooling

        halt_logits = self.halt_head(h).squeeze(-1)  # (B,)
        halt_prob = torch.sigmoid(halt_logits)

        still_running = ~state.halted

        # check halting condition
        should_halt = (state.halt_prob_accum + halt_prob >= self.halt_threshold)

        new_halted = should_halt & still_running

        # compute remainders
        remainders = torch.where(
            new_halted,
            1.0 - state.halt_prob_accum,
            state.remainders
        )

        # compute step weight
        p = torch.where(
            new_halted,
            remainders,
            torch.where(still_running, halt_prob, torch.zeros_like(halt_prob))
        )

        # expand to sequence
        p_expanded = p.view(-1, 1, 1)

        output_accum = state.output_accum + p_expanded * x

        halt_prob_accum = torch.where(
            still_running & ~new_halted,
            state.halt_prob_accum + halt_prob,
            state.halt_prob_accum
        )

        n_updates = state.n_updates + still_running.float()
        halted = state.halted | new_halted

        new_state = SentenceACTState(
            halted=halted,
            halt_prob_accum=halt_prob_accum,
            remainders=remainders,
            n_updates=n_updates,
            output_accum=output_accum,
        )

        return (
            new_state,
            halted.all().item(),
            halt_logits.unsqueeze(-1),   # (B, 1)
            halt_prob.unsqueeze(-1),    # (B, 1)
            should_halt.unsqueeze(-1)       # (B, 1)
        )

    def finalize(
        self,
        x: torch.Tensor,
        state: SentenceACTState
    ):
        still_running = ~state.halted

        remainders = torch.where(
            still_running,
            1.0 - state.halt_prob_accum,
            state.remainders
        )

        p_expanded = remainders.view(-1, 1, 1)

        output = state.output_accum + (still_running.float().view(-1,1,1) * p_expanded) * x

        ponder_cost = (state.n_updates + remainders).mean()

        return output, ponder_cost
    
class BinaryACTHalting(nn.Module):
    def __init__(self, hidden_dim, halt_threshold=0.9):
        super().__init__()
        self.halt_head = nn.Linear(hidden_dim, 1)
        self.halt_threshold = halt_threshold

    def forward(self, x):
        h = x.mean(dim=1)
        logits = self.halt_head(h).squeeze(-1)
        prob = torch.sigmoid(logits)
        return prob, logits

    def should_halt(self, prob):
        return prob.mean() > self.halt_threshold

class ReasoningModule(nn.Module):
    def __init__(self, config: TRLMConfig):
        super().__init__()

        # Projects the concatenated x, y, and z states back to hidden_dim.
        self.combiner = nn.Linear(config.hidden_dim * 3, config.hidden_dim)

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
                transformer_dropout=config.transformer_dropout
            ) for _ in range(config.num_layers)
        ])

        self.num_recursions = getattr(config, 'num_recursions', 3)

        self.gate = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.rms_norm_eps)

        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(-2.0) 

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size, seq_len, hidden_dim = x.shape

        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        n = self.num_recursions

        # Initialize z state to zeros if this is the very first step
        if z is None:
            z = torch.zeros_like(x, device=x.device, dtype=x.dtype)

        # If 'y' is a single pooled state (Batch, Dim), broadcast it to sequence length
        if y.dim() == 2:
            y = y.unsqueeze(1).expand(-1, seq_len, -1)

        # Latent Recursion Loop: refine z 'n' times
        for _ in range(n):
            # Combine Input (x), Answer State (y), and Reasoning State (z)
            combined = torch.cat([x, y, z], dim=-1)     # [Batch, SeqLen, HiddenDim * 3]
            hidden_state = self.combiner(combined)      # [Batch, SeqLen, HiddenDim]
            
            # Pass through the transformer stack
            for layer in self.layers:
                hidden_state = layer(hidden_state, position_ids=position_ids)
                
            # Residual update: add the network's output to the existing z state.
            # z = z + hidden_state

            # Apply gating to prevents explosion in deep recursion
            gate = torch.sigmoid(self.gate(hidden_state))   # (B, T, C)

            # Residual update
            z = z + gate * hidden_state
            
        return z

class TRLM(nn.Module):
    def __init__(self, config: TRLMConfig):
        super().__init__()

        self.config = config
        self.mask_token_id = config.mask_token_id

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.halting = BinaryACTHalting(hidden_dim=config.hidden_dim, halt_threshold=config.halt_threshold)

        self.reasoning_module = ReasoningModule(config)

        self.answer_combiner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.apply(self._init_weights)

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
    
    def _update_answer_state(self, y, z):
        """Updates the answer state 'y' using the reasoning state 'z'."""
        combined = torch.cat([y, z], dim=-1)           # [B, S, H*2]
        h = self.answer_combiner(combined)             # [B, S, H]

        seq_len = h.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=h.device).unsqueeze(0)
        
        # Pass through the shared transformer blocks to leverage shared weights
        for layer in self.reasoning_module.layers:
            h = layer(h, position_ids=position_ids)
            
        return y + h  # Residual connection for stability
    
    def _compute_step_loss(
            self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            halt_logits: torch.Tensor
        ):

        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100 
        )

        # Q-Loss (Halting): We only calculate accuracy on the masked positions
        mask_positions = (labels != -100)
        if mask_positions.any():
            correct_predictions = (logits.argmax(-1)[mask_positions] == labels[mask_positions]).float()
            is_correct = correct_predictions.mean()
                    
            q_loss = F.binary_cross_entropy_with_logits(
                halt_logits, is_correct.expand_as(halt_logits)
            )
        
        step_total_loss = lm_loss + (self.config.q_loss_weight * q_loss)
        
        return lm_loss, q_loss, step_total_loss

    def forward(self, masked_input_ids, labels=None):
        x, y, z = self._initialize_states(masked_input_ids)

        accumulated_loss = 0.0
        accumulated_lm_loss = 0.0
        accumulated_q_loss = 0.0
        final_logits = None
        recursion_depth = 0

        for step in range(self.config.max_supervision_steps):
            recursion_depth = step + 1
            
            # Phase 1: Exploration (T-1 iterations without gradients)
            with torch.no_grad():
                for _ in range(self.config.thinking_steps):
                    z = self.reasoning_module(x, y, z)
                    y = self._update_answer_state(y, z)

            # Phase 2: Supervision (1 final recursion with gradients)
            z = self.reasoning_module(x, y, z)
            y = self._update_answer_state(y, z)

            normalized_y = self.final_norm(y)

            logits = self.lm_head(normalized_y)
            prob, halt_logits = self.halting(normalized_y)
            final_logits = logits

            if labels is not None:
                step_lm, step_q, step_total = self._compute_step_loss(
                    logits=logits, labels=labels, halt_logits=halt_logits
                )
                
                accumulated_loss += step_total
                accumulated_lm_loss += step_lm
                accumulated_q_loss += step_q

            # Early Stopping Check
            if self.halting.should_halt(prob):
                break

            y = y.detach()
            z = z.detach()

        avg_lm_loss = accumulated_lm_loss / recursion_depth if labels is not None else 0.0
        avg_q_loss = accumulated_q_loss / recursion_depth if labels is not None else 0.0
        avg_total_loss = accumulated_loss / recursion_depth if labels is not None else 0.0
        
        return final_logits, avg_total_loss, avg_lm_loss, avg_q_loss, recursion_depth, prob
        
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, max_refinement_steps=10):
        """
        prompt_ids: [Batch, PromptSeqLen] - The user's input.
        """

        batch_size = prompt_ids.size(0)
        device = prompt_ids.device

        # Append [MASK] tokens to the end of the prompt to act as placeholders for the answer
        mask_tokens = torch.full((batch_size, max_new_tokens), self.mask_token_id, device=device, dtype=torch.long)
        draft_ids = torch.cat([prompt_ids, mask_tokens], dim=1)
        
        # Initialize continuous states
        x, y, z = self._initialize_states(draft_ids)

        # Iteratively Refine the Draft
        for step in range(max_refinement_steps):
            
            # Deep reasoning pass
            z = self.reasoning_module(x, y, z)
            y = self._update_answer_state(y, z)
            
            logits = self.lm_head(y)
            prob, _ = self.halting(y)
            
            # If the halting mechanism is confident, we can stop editing early
            if self.halting.should_halt(prob):
                break

        # Finalize the Output
        # Extract the final predicted token IDs
        final_token_ids = logits.argmax(dim=-1)
        
        # We only care about the newly generated part (the masked portion)
        generated_answers = final_token_ids[:, prompt_ids.size(1):]
        
        return generated_answers
        
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
