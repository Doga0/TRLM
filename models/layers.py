import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

from models.kernels.swiglu import SiLUMulFunction
from models.kernels.rope import rope_backward, rope_forward
from models.kernels.utils import ensure_contiguous
from models.kernels.rms_norm import rms_norm_forward, rms_norm_backward
from models.utils import trunc_normal_init_

# Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py#L204
class RotaryEmbedding(torch.autograd.Function):
    """
    Triton implementation of the Rotary Positional Embedding (RoPE) operation. Please note that
    this implements the HuggingFace Llama & Mistral version, whose rotation matrix is slightly different
    than the original RoPE paper.

    Please find the corresponding HuggingFace implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

    For more details about the rotation matrix used here, please refer to:
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """

        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None, None, None
    
class RotaryFrequencyGenerator(nn.Module):
    """Generates cos and sin frequencies for RoPE."""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calculate inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: [batch_size, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute frequencies
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        
        # Concat to get full head dimension [batch_size, seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        
        return cos, sin
    
# Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_size, hidden_act):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        if hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(SiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
    
#Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py#L583
class RMSNormFunction(torch.autograd.Function):
    """
    Performs RMSNorm (Root Mean Square Normalization), which normalizes the input tensor `X` using the
    weight tensor `W`, with an optional offset and casting mode.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    In addition, different models cast their inputs at different places during RMSNorm computation. For
    example, Gemma casts everything to fp32 nefore starting the computation, while Llama casts only the
    inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
    support the following casting modes (they match HuggingFace Transformers' implementations):
    - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
    - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
    - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

    `in_place` option means whether to in_place modify dY to store dX. This is default to `True` to save memory. However, under certain cases, it can produce incorrect inputs.
        For example, gemma2 uses two rmsnorm sequentially with residual in between. The resesidual part needs dY so it cannot be modified in-place.
        Therefore, for the patching of RMSNorm in gemma2, we set `in_place` to `False`
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        if isinstance(X, torch.distributed.tensor.DTensor):
            # Input tensor is output of a tensor parallel module and
            # needs to be gathered to a local tensor to compute
            # RMSE layer norm on each TP worker.
            # TODO: support CP.
            X = X.full_tensor()

        Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = rms_norm_forward(X, W, eps, offset, casting_mode, row_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.elementwise_affine = W is not None
        if W is not None:
            ctx.save_for_backward(X, W, RSTD)
        else:
            ctx.save_for_backward(X, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        if ctx.elementwise_affine:
            X, W, RSTD = ctx.saved_tensors
        else:
            X, RSTD = ctx.saved_tensors
            W = None

        if isinstance(dY, torch.distributed.tensor.DTensor):
            # Gradients are output of a tensor parallel module and
            # needs to be gathered to a local tensor for computing RMSE layer.
            # TODO: support CP.
            dY = dY.full_tensor()

        dX, dW = rms_norm_backward(
            dY, X, W, RSTD, ctx.offset, ctx.casting_mode, ctx.BLOCK_SIZE, ctx.num_warps, ctx.in_place, ctx.row_mode
        )
        return dX, dW, None, None, None, None, None
    
class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return RMSNormFunction.apply(
            hidden_states, 
            self.weight, 
            self.variance_epsilon,
            0.0,      # offset (Gemma uses 1.0, Llama uses 0.0)
            "llama",  # casting_mode
            True      # in_place
        )
    
class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
