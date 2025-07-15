import os
import torch
from torch.nn import functional as F
from qwen3_model import Qwen3, GPTConfig
from tokenizers import ByteLevelBPETokenizer
from sft_dataset import get_sft_dataloader
from util import load_config

def main():
    # Load config
    GPT_CONFIG, HYPERS, FILES, TRAINING = load_config(config_file="config.txt")
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    
    # Load tokenizer
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]
    tokenizer = ByteLevelBPETokenizer(
        FILES["tokenizer_dir"] + "gabgpt-vocab.json",
        FILES["tokenizer_dir"] + "gabgpt-merges.txt"
    )
    tokenizer.add_special_tokens(special_tokens)
    
    # Load pretrained model
    config = GPTConfig(
        n_head=GPT_CONFIG["n_head"],
        n_embd=GPT_CONFIG["n_embd"], 
        block_size=GPT_CONFIG["block_size"],
        n_layer=GPT_CONFIG["n_layer"],
        vocab_size=GPT_CONFIG["vocab_size"],
        dropout=GPT_CONFIG["dropout"]
    )
    
    model = Qwen3(config)
    
    # Load pretrained weights
    checkpoint = torch.load("checkpoints/ckpt-final.pt", map_location=device)
    
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Setup optimizer with lower learning rate for fine-tuning
    optimizer = model.configure_optimizers(
        weight_decay=HYPERS["weight_decay"],
        learning_rate=HYPERS["learning_rate"] * 0.1,  # Lower LR for SFT
        betas=(HYPERS["beta1"], HYPERS["beta2"]),
        device_type=device
    )
    
    # Get dataloader
    dataloader = get_sft_dataloader(tokenizer, batch_size=2, max_length=512)
    
    # Training loop
    model.train()
    for epoch in range(3):  # Few epochs for SFT
        total_loss = 0
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_ids)  # Don't use model's loss
            
            # Compute masked loss manually
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=-100  # Ignore masked tokens
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} completed, Average Loss: {total_loss/len(dataloader):.4f}")
        
        # Save checkpoint
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, f"checkpoints/sft-epoch-{epoch}.pt")

if __name__ == "__main__":
    main()