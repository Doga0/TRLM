import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from models.layers import *


class Attention(nn.Module):
    """
    Multi-head self-attention with rotary position embeddings.

    Implements scaled dot-product attention with:
    - Separate Q, K, V projections (no bias)
    - Rotary position embeddings applied to Q and K
    - Multi-head attention with efficient F.scaled_dot_product_attention
    - Output projection (no bias)

    Args:
        hidden_dim: Model dimension (default 512)
        num_heads: Number of attention heads (default 8)
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, max_seq_len: int = 1024):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q, K, V projections - no bias per paper
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection - no bias per paper
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rope_generator = RotaryFrequencyGenerator(
            dim=self.head_dim, 
            max_position_embeddings=max_seq_len
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids,
        mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (B, seq_len, hidden_dim)
            mask: Optional attention mask of shape (B, seq_len) or (B, seq_len, seq_len)
                  where True/1 indicates valid positions
            positions: Optional position indices of shape (B, seq_len)
                      If None, uses simple sequential positions

        Returns:
            Output tensor of shape (B, seq_len, hidden_dim)
        """
        B, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, seq_len, hidden_dim)
        k = self.k_proj(x)  # (B, seq_len, hidden_dim)
        v = self.v_proj(x)  # (B, seq_len, hidden_dim)

        # Reshape to multi-head: (B, seq_len, hidden_dim) -> (B, seq_len, num_heads, head_dim)
        q = q.view(B, seq_len, self.num_heads, self.head_dim)
        k = k.view(B, seq_len, self.num_heads, self.head_dim)
        v = v.view(B, seq_len, self.num_heads, self.head_dim)

        # Apply rotary position embeddings to Q and K
        if positions is None:
            # Use sequential positions if not provided
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)

        cos, sin = self.rope_generator(x, position_ids)
        
        # Transpose to (B, num_heads, seq_len, head_dim) for attention
        q = q.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, seq_len, head_dim)

        q, k = RotaryEmbedding.apply(q, k, cos, sin)

        # Prepare attention mask for scaled_dot_product_attention
        # F.scaled_dot_product_attention expects mask where True means attend
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # (B, seq_len) -> (B, 1, 1, seq_len) for broadcasting
                # This masks out invalid keys
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (B, seq_len, seq_len) -> (B, 1, seq_len, seq_len)
                attn_mask = mask.unsqueeze(1)

        # Scaled dot-product attention
        # Output: (B, num_heads, seq_len, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Transpose back and reshape: (B, num_heads, seq_len, head_dim) -> (B, seq_len, hidden_dim)
        out = out.transpose(1, 2).contiguous()

        out = out.reshape(B, seq_len, self.hidden_dim)

        return self.o_proj(out)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups, max_seq_len: int = 1024, dropout = None):
        super().__init__()
        self.dropout = dropout

        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

        self.d_model = d_model
        self.num_heads = num_heads    
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_model)

        self.rope_generator = RotaryFrequencyGenerator(
            dim=self.head_dim, 
            max_position_embeddings=max_seq_len
        )

    def forward(self, x: torch.Tensor, position_ids, mask: torch.Tensor = None, is_causal: bool = False):
        batch_size, seq_length, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_groups, self.head_dim).transpose(1, 2)

        cos, sin = self.rope_generator(x, position_ids)
        q, k = RotaryEmbedding.apply(q, k, cos, sin)

        if self.num_groups != self.num_heads:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=0.0 if not self.training else self.dropout,
            is_causal=is_causal
        )
        
        out = out.transpose(1, 2).reshape(batch_size, seq_length, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 rms_norm_eps, 
                 attn_type,
                 num_heads,
                 max_seq_len,
                 num_groups,
                 swiglu_intermediate_size,
                 swiglu_hidden_act,
                 attn_dropout,
                 transformer_dropout,
                 use_adaln: bool = False
            ):
        super().__init__()

        self.use_adaln = use_adaln

        self.norm1 = RMSNorm( 
            hidden_dim=hidden_dim, 
            eps=rms_norm_eps
        ) 

        self.attn = None

        if attn_type == "mha":
            self.attn = Attention( 
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len
            )

        elif attn_type == "gqa":
            self.attn = GroupedQueryAttention( 
                d_model=hidden_dim,
                num_heads=num_heads,
                num_groups=num_groups,
                max_seq_len=max_seq_len,
                dropout=attn_dropout
            )

        self.mlp = SwiGLU(
            hidden_dim=hidden_dim, 
            intermediate_size=swiglu_intermediate_size,
            hidden_act=swiglu_hidden_act
        )
        
        self.norm2 = RMSNorm( 
            hidden_dim=hidden_dim, 
            eps=rms_norm_eps
        ) 

        self.dropout = nn.Dropout(transformer_dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if cond is not None and self.use_adaln:
            gate_msa, gate_mlp, scale_msa, scale_mlp = cond.chunk(4, dim=-1)
            gate_msa = gate_msa.unsqueeze(1)   # [B, 1, H]
            gate_mlp = gate_mlp.unsqueeze(1)
            scale_msa = scale_msa.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
        else:
            # Normal residual (adaLN yoksa)
            gate_msa = gate_mlp = 1.0
            scale_msa = scale_mlp = 0.0
        
        # Pre-Norm
        residual = x
        x = self.norm1(x)
        if self.use_adaln:
            x = x * (1 + scale_msa)  
        x = self.attn(
            x=x, 
            position_ids=position_ids, 
            mask=mask, 
            is_causal=is_causal
        )
        if self.use_adaln:
            x = gate_msa * x         
        x = residual + self.dropout(x)
        
        residual = x
        x = self.norm2(x)
        if self.use_adaln:
            x = x * (1 + scale_mlp)  
        x = self.mlp(x)
        if self.use_adaln:
            x = gate_mlp * x         
        x = residual + self.dropout(x)
        
        return x

