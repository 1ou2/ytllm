import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        # dim: dimension of the embedding vectors
        # max_seq_len: maximum sequence length the model can handle
        super().__init__()
        
        # Check that dimension is even, required for RoPE
        if dim % 2 != 0:
            raise ValueError(f"Dimension for RotaryPositionalEmbedding must be even, got {dim}")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        
         # Calculate frequency bands using the standard formula from the paper
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # Create position indices from 0 to max_seq_len-1
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        # Outer product creates a matrix where each position has all frequency components
        freqs = torch.outer(t, inv_freq)
        # Duplicate frequencies to match embedding dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        # Pre-compute and cache sin and cos values for efficiency
        # [1, 1, max_seq_len, dim] shape allows for broadcasting
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def _apply_rotary(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos = self.cos_cached[:, :, :seq_len, ...].to(x.dtype)
        sin = self.sin_cached[:, :, :seq_len, ...].to(x.dtype)
        # In 2D space, rotating a vector (x, y) by angle θ gives:
        # x' = x·cos(θ) - y·sin(θ)
        # y' = x·sin(θ) + y·cos(θ)
        # The _rotate_half function implements this rotation by treating consecutive pairs of dimensions as 2D planes. 
        # When we split the embedding into two halves (x1, x2) and create (-x2, x1), 
        # we're setting up the components needed for the rotation formula
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Split the embedding into two halves along the last dimension
        x1, x2 = x.chunk(2, dim=-1)
        # Create a new tensor by concatenating -x2 and x1
        # This effectively rotates each pair of dimensions
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        q = self._apply_rotary(q, seq_len)
        k = self._apply_rotary(k, seq_len)
        return q, k

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.1, max_seq_len: int = 4096):
        super().__init__()
        # Validate head_dim is even for RoPE
        if head_dim % 2 != 0:
            raise ValueError(f"Head dimension must be even for RotaryPositionalEmbedding, got {head_dim}")
            
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        assert num_heads * head_dim == dim, "num_heads * head_dim must equal model dim"

        self.qkv_proj = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryPositionalEmbedding(head_dim, max_seq_len)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, k = self.rotary(q, k)

        # --- Build causal mask ---
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )  # [seq_len, seq_len], True above diagonal
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.to(x.dtype) * torch.finfo(x.dtype).min  # 0 for allowed, -inf for masked

        # --- Build padding mask ---
        if attention_mask is not None:
            # attention_mask: [batch, 1, 1, seq_len] with 0 for pad, 1 for real
            # Already processed in GPTModel, so just use as is
            mask = attention_mask + causal_mask  # Broadcasting: [batch, 1, 1, seq_len] + [1, 1, seq_len, seq_len]
        else:
            mask = causal_mask  # [1, 1, seq_len, seq_len]

        # --- Use PyTorch's efficient attention if available ---
        if torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False  # Mask already includes causality
            )
        else:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = attn + mask  # Additive mask: -inf for masked positions
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn) if self.training else attn
            x = attn @ v

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, ff_dim: int, dropout: float = 0.1, max_seq_len: int = 4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, head_dim, dropout, max_seq_len)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim, dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ff(self.norm2(x))
        return x

class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        head_dim: int = 64,
        ff_dim: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, head_dim, ff_dim, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        #  # Tie weights: output projection uses the same weights as the embedding
        self.head.weight = self.token_emb.weight
 
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.token_emb(input_ids)

        # Process attention mask if provided (for handling padding)
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] with 1 for real, 0 for pad
            if attention_mask.dim() == 2:
                # Expand to [batch, 1, 1, seq_len] for broadcasting in attention
                attention_mask = attention_mask[:, None, None, :]
            # Convert to additive mask: 0 for real, -inf for pad
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        return self.head(x)
