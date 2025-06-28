import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import torch

# ----------------------------------------------------------------------
# GPT Model
# ----------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 32768 # number of tokens: custom tokenizer used
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    num_kv_groups:int = 4  # number of groups in Group Query Attention
    n_embd: int = 768 # embedding dimension
    dropout: float = 0.1 # dropout rate

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


class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.num_kv_groups = config.num_kv_groups
        self.group_size = self.n_head // config.num_kv_groups
        self.dropout_rate = config.dropout

        assert config.n_head % config.num_kv_groups == 0, f"n_head ({config.n_head}) must be divisible by num_kv_groups ({config.num_kv_groups})"

        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.num_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.num_kv_groups * self.head_dim, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # we will scale the standard deviation intialization by 1 / sqrt(n_head)
        # this is basically a flag for the weights_init function
        self.c_proj.SCALE_INIT = 1
        self.rotary = RotaryPositionalEmbedding(self.head_dim, config.block_size)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # compute Q,K,V in parallel
        # project the query, key, and value vectors
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape Q: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Reshape K, V: (B, T, num_kv_groups, head_dim) -> (B, num_kv_groups, T, head_dim)
        k = k.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # apply rotary embeddings
        q, k = self.rotary(q, k)

        # Expand K and V to match the number of query heads for flash attention
        # repeat_interleave replicates each KV head group_size times
        k_expanded = k.repeat_interleave(self.group_size, dim=1)  # (B, n_head, T, head_dim)
        v_expanded = v.repeat_interleave(self.group_size, dim=1)  # (B, n_head, T, head_dim)

        y = F.scaled_dot_product_attention(q, k_expanded, v_expanded, dropout_p=self.dropout_rate if self.training else 0,is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return y
    

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        # epislon = small value to ensure we do not divide by 0
        self.eps = eps
        # learnable scaling factor, the network will learn how to scale each parameter
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # save current type
        input_dtype = x.dtype
        # cast to float32, to ensure max precision as we are using pow(2)
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        # scale the norm using the learned parameters
        norm_x = norm_x * self.scale

        return norm_x.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU requires two parallel linear layers for the gating mechanism
        # Each needs to project to the intermediate dimension

        intermediate_size = int(4 * config.n_embd * 2/3)  # Reduce size to compensate for 2 layers
        self.c_fc1 = nn.Linear(config.n_embd, intermediate_size)
        self.c_fc2 = nn.Linear(config.n_embd, intermediate_size)
        
        # Output projection back to original dimension
        self.c_proj = nn.Linear(intermediate_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Keep the scale initialization for the output projection
        self.c_proj.SCALE_INIT = 1
    
    def forward(self, x):
        # SwiGLU: SiLU(W1 * x) * (W2 * x)
        x_fc1 = self.c_fc1(x)  # First linear transformation
        x_fc2 = self.c_fc2(x)  # Second linear transformation (gate)
        
        # Apply SiLU to first path and multiply by second path
        x = nn.functional.silu(x_fc1) * x_fc2
        
        # Project back to original dimension
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = GroupQueryAttention(config)
        self.norm_2 = RMSNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.ff(self.norm_2(x))
        return x


class Qwen3(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop_emb = nn.Dropout(config.dropout)
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # standard deviation
        # ~ 0.03 for n_embed = 768
        std = self.config.n_embd ** -0.5
        if isinstance(module, nn.Linear):
            # because of shortcut connection, the gradient will increase more for some layers            
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = self.drop_emb(x)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.final_norm(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=True)
        return optimizer
  