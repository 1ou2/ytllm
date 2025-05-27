
from model_gpt import GPTModel
from dataloader import DataLoaderLite
from transformers import get_linear_schedule_with_warmup
import torch
import tiktoken
import torch.nn.functional as F
import time
import os


GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "embed_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG2 = {
    "vocab_size": 32768,
    "context_length": 1024,
    "embed_dim": 1024,
    "n_layers": 16,
    "n_heads": 16,
    "drop_rate": 0.1,
    "qkv_bias": False
    }

def generate_text_completion(model, text):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    #encoded = [2132, 5555, 21455]
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


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Load the model, optimizer, scheduler, from a file.
    Returns (epoch, step, loss, train_loader_state,stats)
    Raises ValueError in case of error
    """
    checkpoint = torch.load(path,map_location=device,weights_only=True)
            
    # Create a new state dict removing '_orig_mod' prefix from checkpoint
    # when using torch.compile with aot_eager backend, the keys have a prefix '_orig_mod.'
    # we need, to remov this prefix to be able to load the model otherwise we have a mismatch
    fixed_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('_orig_mod.'):
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
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss'], checkpoint['train_loader_state']

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

# -------------------------------------------------------------



device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

device_type = "cuda" if device.startswith("cuda") else "cpu"

# trying to not get out of memory error
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"


tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG)
model.to(device)


checkpoint = get_last_checkpoint("checkpoints")
if checkpoint is not None:
    start_epoch, start_step, loss, train_state = load_checkpoint(checkpoint, model, None, None,device=device)
    print(f"Loaded checkpoint: epoch {start_epoch}, step {start_step}, loss {loss}")
else:
    print("No checkpoint found")



t0 = time.time()
text = generate_text_completion(model, "Je suis")
t1 = time.time()

print(f"Time: {t1-t0}")
print(f"Generated text: ///{text}///")