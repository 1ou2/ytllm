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


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # shortcut connection will be added
        # we will scale the standard deviation intialization by 1 / sqrt(n_head)
        # this is basically a flag for the weights_init function
        self.c_proj.SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_rate = config.dropout
        # dropout of residual connection
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryPositionalEmbedding(config.n_embd // config.n_head, config.block_size)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply rotary embeddings
        q, k = self.rotary(q, k)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_rate if self.training else 0,is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop_emb = nn.Dropout(config.dropout)
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.ln_f = nn.LayerNorm(config.n_embd)
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
        x = self.ln_f(x)
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
  