
from model_gpt import GPTModel
from dataloader import DataLoaderLite
from transformers import get_linear_schedule_with_warmup
from tokenizers import ByteLevelBPETokenizer
import torch
import tiktoken
import torch.nn.functional as F
import time
import os
import configparser
import ast

def load_config(config_file="config.txt"):
    config = configparser.ConfigParser()
    config.read(config_file)
    # Convert string values to appropriate Python types
    def parse_value(value):
        try:
            # Try to evaluate as literal (for boolean, None, etc)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If it fails, return as string
            return value

    # Create configuration dictionaries
    GPT_CONFIG = {
        key: parse_value(value)
        for key, value in config['model'].items()
    }
    
    HYPERS = {
        key: parse_value(value)
        for key, value in config['hypers'].items()
    }
    
    #
    FILES = {
        key: parse_value(value)
        for key, value in config['files'].items()
    }

    TRAINING = {
        key: parse_value(value)
        for key, value in config['training'].items()
    }
    
    return GPT_CONFIG, HYPERS, FILES, TRAINING


def generate_text_completion(model, text):
    if tokenizerType == "custom":
        encoded = tokenizer.encode(text).ids
    elif tokenizerType == "gpt2":
        encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizerType}")

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

def save_checkpoint(model, optimizer, scheduler, train_loader, config, epoch, step, loss, save_dir):
    """
    Save the model, optimizer, scheduler, and epoch to a file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/checkpoint_{epoch}_{step:07d}_{loss:.2f}.pt"
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config,
        'train_loader_state': train_loader.get_state()
    }
    torch.save(checkpoint, path)
    log_print(f"Saved checkpoint to {path}")


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

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # If using GPU, move optimizer states to GPU
        if device == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    else:
        raise ValueError("Optimizer state dict not found in checkpoint. Unable to load optimizer.")
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        raise ValueError("Scheduler state dict not found in checkpoint. Unable to load scheduler.")
    
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

def log_print(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


GPT_CONFIG, HYPERS, FILES, TRAINING = load_config()

device = "cpu"
backend = "inductor"
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(current_device)
    if major < 7:
        backend = "aot_eager"
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

device_type = "cuda" if device.startswith("cuda") else "cpu"

# trying to not get out of memory error
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

log_file = open(FILES["log_file"], "a")
stats_file = open(FILES["stat_file"], "a")

# Write header for stats file (if it's empty)
if stats_file.tell() == 0:
    stats_file.write("epoch,step,loss,learning_rate\n")

log_print("------------------------------------------------------------------------------------")
# print current date YYYY-MM-DD-HH-MM-SS
log_print(f"Date - time : {time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
log_print(f"using device: {device}")
log_print(f"using backend: {backend}")

B = TRAINING["batch_size"]
T = GPT_CONFIG["context_length"]
start_epoch = 0
start_step = 0

useCheckpoint = TRAINING["use_checkpoint"]
tokenizerType = GPT_CONFIG["tokenizer"]
valid_token_dir = FILES["token_dir"] + "valid/"
train_token_dir = FILES["token_dir"] + "train/"
val_loader = DataLoaderLite(B, T, split="valid",token_dir=valid_token_dir)
train_loader = DataLoaderLite(B, T, "train", token_dir=train_token_dir)

if tokenizerType == "custom":
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        FILES["tokenizer_dir"] + "gabgpt-vocab.json",
        FILES["tokenizer_dir"] + "gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)
elif tokenizerType == "gpt2":
    tokenizer = tiktoken.get_encoding("gpt2")

# number of batches in one epoch
nb_train_steps = int(HYPERS["tokens_per_shard"] / (B * T)) * len(train_loader.shards)
nb_val_steps = int(len(val_loader.shards) * HYPERS["tokens_per_shard"] / (B * T))
log_print(f"nb_train_steps: {nb_train_steps}")
log_print(f"nb_val_steps: {nb_val_steps}")

total_steps = nb_train_steps * HYPERS["epochs"]
warmup_steps = total_steps // 10 # use 10% for warmup

model = GPTModel(GPT_CONFIG)
model.to(device)
#optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERS["lr"], betas=(HYPERS["beta1"], HYPERS["beta2"]), eps=HYPERS["eps"], weight_decay=HYPERS["weight_decay"])
optimizer = torch.optim.AdamW(model.parameters(), lr=HYPERS["lr"])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

if useCheckpoint:
    checkpoint = get_last_checkpoint(FILES["checkpoint_dir"])
    if checkpoint is not None:
        start_epoch, start_step, loss, train_state = load_checkpoint(checkpoint, model, optimizer, scheduler,device=device)
        shard_index = train_state["shard_index"]
        log_print(f"Loaded checkpoint: epoch {start_epoch}, step {start_step}, loss {loss} - shard: {shard_index}")
        train_loader.set_state(train_state)

model = torch.compile(model, backend=backend)

t0 = time.time()
for epoch in range(start_epoch,HYPERS["epochs"]):
    for step in range(start_step, nb_train_steps):
        if step > start_step + TRAINING["max_steps"]:
            break
        model.train()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, GPT_CONFIG["vocab_size"]), y.view(-1))
        loss.backward()  # backward pass
        optimizer.step() # update weights
        scheduler.step() # update learning rate
        optimizer.zero_grad() # reset gradients
        
        if step % TRAINING["log_interval"] == 0:
            log_msg = f"step: {step}, loss: {loss.item()}"
            log_print(log_msg)  
            
            # Write stats in CSV format for easy parsing
            stats_msg = f"{epoch},{step},{loss.item()},{scheduler.get_last_lr()[0]}\n"
            stats_file.write(stats_msg)
            stats_file.flush()
        
        if step % TRAINING["gen_text_interval"] == 0:
            text = generate_text_completion(model, "Je suis")
            log_msg = f"{step}: Generated text: ///{text.strip()}///"
            log_print(log_msg)
        # evaluate validation loss
        if step % TRAINING["eval_interval"] == 0 and step > 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                # average loss 
                val_loss_steps = TRAINING["validation_steps"]
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, GPT_CONFIG["vocab_size"]), y.view(-1))
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                log_print(f"Validation loss: {val_loss_accum.item()}")
        if step %  TRAINING["checkpoint_interval"]  == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, train_loader, GPT_CONFIG, epoch, step, loss.item(), FILES["checkpoint_dir"])

save_checkpoint(model, optimizer, scheduler, train_loader, GPT_CONFIG, epoch, step, loss.item(), FILES["checkpoint_dir"])

t1 = time.time()
wrapup_message = f"""
Time: {t1-t0}
Processed Tokens: {B}*{T}*{step - start_step} = {B*T*(step - start_step)}
Tokens/s: {(B*T*(step - start_step))/(t1-t0)}
Loss: {loss.item()}
Total Tokens: {B}*{T}*{step} = {B*T*step}
Shard index: {train_loader.current_shard_index}
"""
log_print(wrapup_message)
log_file.close()
stats_file.close()

