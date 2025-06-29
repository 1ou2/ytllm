import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from datetime import datetime
from qwen3_model import Qwen3, GPTConfig
    
def load_parameters(path,device="cpu"):
    """
    Load the model, optimizer, scheduler, from a file.
    Returns (epoch, step, loss, train_loader_state, stats)
    Raises ValueError in case of error
    """
    checkpoint = torch.load(path, map_location=device,weights_only=True)
    return checkpoint['train_loader_state'], checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """
    Load the model, optimizer, scheduler, from a file.
    Returns (epoch, step, loss, train_loader_state,stats)
    Raises ValueError in case of error
    """
    checkpoint = torch.load(path,map_location=device,weights_only=True)
    # Create a new state dict removing '_orig_mod' prefix from checkpoint
    # when using torch.compile with aot_eager backend, the keys have a prefix '_orig_mod.module.'
    # we need, to remov this prefix to be able to load the model otherwise we have a mismatch
    fixed_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('_orig_mod.module.'):
            new_key = key.replace('_orig_mod.module.', '')
            fixed_state_dict[new_key] = value
        elif key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value

    # Compare the shapes of tensors
    for key in fixed_state_dict:
        if key in model.state_dict():
            ckpt_shape = fixed_state_dict[key].shape
            model_shape = model.state_dict()[key].shape
            if ckpt_shape != model_shape:
                print(f"Shape mismatch for {key}: checkpoint {ckpt_shape} vs model {model_shape}")
        else:
            print(f"Key {key} not found in model state dict")

    model.load_state_dict(fixed_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def get_last_checkpoint(save_dir):
    """
    Get the last checkpoint file in the given directory.
    Returns the path to the last checkpoint file.
    """
    # check if directory exists
    if not os.path.exists(save_dir):
        return None

    checkpoints = sorted([f for f in os.listdir(save_dir) if f.startswith("checkpoint_")])
    if len(checkpoints) == 0:
        return None
    return os.path.join(save_dir, checkpoints[-1])


def generate_text_completion(model, text):
    encoded = tokenizer.encode(text).ids

    # model expects an input in batch format
    idx = torch.tensor(encoded).unsqueeze(0).to(device)

    #for _ in range(GPT_CONFIG["context_length"]-len(idx)):
    for _ in range(50):
        model.eval()
        with torch.no_grad():
            # logits shape : (batch_size, sequence_length, vocab_size)
            # usually batch_size is 1 for text generation
            logits = model(idx)
            # Get logits for the last token of the sequence
            # shape : (batch_size, vocab_size)
            logits = logits[:, -1, :]
            # shape : (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    
    return tokenizer.decode(idx.squeeze().tolist())


from util import load_config, LogPrinter

def get_device():
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    device_name = device
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)  # 0 is the GPU index
    return device

def get_tokenizer(tokenizer_dir):
    from tokenizers import ByteLevelBPETokenizer
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        tokenizer_dir + "gabgpt-vocab.json",
        tokenizer_dir + "gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def evaluate(model):
    B = TRAINING["micro_batch_size"]
    T = GPT_CONFIG["context_length"]
    # Token loader
    valid_token_dir = FILES["token_dir"] + "valid/"
    nb_shards = TRAINING["n_shards"]
    from dataloader import IndexedDataLoader
    import random
    val_loader = IndexedDataLoader(B, T, split="valid", nb_shards=nb_shards, token_dir=valid_token_dir)
    val_loader.reset()
    val_loader.current_token_index=random.choice(range(TRAINING["batch_size"]))
    print(val_loader.current_shard_index)
    model.eval()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()

    print(f"validation loss: {val_loss_accum.item():.4f} |")

def generate_text(model,tokenizer,text, max_length):
    model.eval()
    num_return_sequences = TRAINING["num_generation_sequences"]
    tokens = tokenizer.encode(text).ids
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            
            logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        print(f"---\n{decoded}\n---")
        #print(f"{tokens=}")

def load_model(path):
    # 1. create model
    #model = GPT(GPTConfig())
    config = GPTConfig(n_head=GPT_CONFIG["n_head"],n_embd=GPT_CONFIG["n_embd"],block_size=GPT_CONFIG["block_size"],
                    n_layer=GPT_CONFIG["n_layer"],vocab_size=GPT_CONFIG["vocab_size"],dropout=GPT_CONFIG["dropout"])
    # 1. create model
    model = Qwen3(config)
    # 2. move to the correct GPU
    model.to(device)

    # 3. load checkpoint
    checkpoint = get_last_checkpoint(path)
    
    if checkpoint is not None:
        load_checkpoint(checkpoint, model, device=device)

    return model

if __name__ == "__main__":
    device = get_device()
    GPT_CONFIG, HYPERS, FILES, TRAINING = load_config()

    # Add special tokens to the loaded tokenizer
    tokenizer = get_tokenizer(FILES["tokenizer_dir"])

    model = load_model(FILES["checkpoint_dir"])
    
    #generate_text(model, tokenizer, "<|endoftext|>",100)
    #print("done")
    generate_text(model, tokenizer, "La Seine est un fleuve qui",100)
    #evaluate(model)






            