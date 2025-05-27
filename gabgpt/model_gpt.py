import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        d_in : dimension of input
        d_out : dimension of output
        context_length : 
        dropout : percentage of neurons to dropout
        num_heads : number of heads
        qkv_bias : use a bias ?
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout
        # check if flash attention is available
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # create an upper triangle mask of ones (excluding diagonal), and register as "mask", can be accessed using self.mask
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):
        """
        x : input batch
        endoftext_mask : Apply custom mask to ignore <|endoftext|> tokens (if provided)
        """
        # batch, number of tokens, input embedding dimension
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # scaled_dot_product_attention is a function from PyTorch that implements Flash Attention

            context_vec = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self.dropout_rate if self.training else 0, is_causal=True)
            context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        else:
            # reshape keys, queries, values
            # we have concatenated all heads in big matrixes
            # we need to unroll and split the d_out dimension into num_heads * head_dim
            # view :(b, num_tokens, d_out) -> (b,num_tokens, num_heads, head_dim)
            # transpose(1,2) : (b,num_tokens, num_heads, head_dim) -> (b,num_heads,num_tokens, head_dim)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

            # compute dot product for each head
            attn_scores = queries @ keys.transpose(2, 3)
            
            # if mask value is true, fill with -infinity
            # [:num_tokens, :num_tokens] : is used to handle smaller input size
            # The syntax [:num_tokens] means "take all elements from the start up to num_tokens". It's equivalent to [0:num_tokens]
            attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))

            d_k = keys.shape[-1]
            attn_weights = torch.nn.functional.softmax(attn_scores / torch.sqrt(torch.tensor(d_k)), dim=-1)
            # dropout some weights to prevent overfitting
            attn_weights = self.dropout(attn_weights)

            # combine heads
            # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
            # transpose(1, 2) : (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
            # contiguous() : to avoid error when using view() later
            # view : (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
    
            context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        # add a linear projection
        # After concatenating all the attention heads together, we need to project the concatenated vector back to the expected output dimension
        # The projection allows the model to learn how to optimally combine and weight the information from different attention heads
        context_vec = self.out_proj(context_vec)

        return context_vec
    


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embed_dim"], 4 * config["embed_dim"]),
            nn.GELU(),
            nn.Linear(4 * config["embed_dim"], config["embed_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(d_in=config["embed_dim"],
                                      d_out=config["embed_dim"],
                                      context_length=config["context_length"],
                                      dropout=config["drop_rate"],
                                      num_heads=config["n_heads"],
                                      qkv_bias=config["qkv_bias"])
        
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["embed_dim"])
        self.norm2 = LayerNorm(config["embed_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # feed forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.pos_emb = nn.Embedding(config["context_length"],config["embed_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
        self.final_norm = LayerNorm(config["embed_dim"])
        self.out_head = nn.Linear(config["embed_dim"], config["vocab_size"],bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        # if the sequence context is greater than or equal to the context window, crop it
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            # logits shape : (batch_size, sequence_length, vocab_size)
            # usually batch_size is 1 for text generation
            logits = model(idx_cond)
            # Get logits for the last token of the sequence
            # shape : (batch_size, vocab_size)
            logits = logits[:, -1, :]
            # shape : (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
