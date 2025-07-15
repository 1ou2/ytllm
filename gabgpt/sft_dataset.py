import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

class AlpacaDataset(Dataset):
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-french")
        self.data = dataset['train']
        
        # Special tokens
        self.user_token = "<|user|>"
        self.bot_token = "<|bot|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|gab1|>"  # Dedicated padding token
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        conversation = self.data[idx]['conversations']
        
        # Format conversation and track positions
        text = ""
        mask_positions = []  # Track where to mask (instructions)
        
        for turn in conversation:
            start_pos = len(text)
            if turn['from'] == 'human':
                text += f"{self.user_token}{turn['value']}"
                # Mark instruction tokens for masking
                mask_positions.append((start_pos, len(text)))
            elif turn['from'] == 'gpt':
                text += f"{self.bot_token}{turn['value']}"
                # Don't mask assistant responses
        text += self.eos_token
        
        # Tokenize
        tokens = self.tokenizer.encode(text).ids
        
        # Create mask for labels (-100 = ignore in loss)
        labels_mask = [False] * len(tokens)  # False = compute loss
        
        # Mark instruction tokens to ignore
        for start_char, end_char in mask_positions:
            start_token = len(self.tokenizer.encode(text[:start_char]).ids)
            end_token = len(self.tokenizer.encode(text[:end_char]).ids)
            for i in range(start_token, min(end_token, len(labels_mask))):
                labels_mask[i] = True  # True = ignore in loss
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            labels_mask = labels_mask[:self.max_length]
        
        # Pad if too short
        pad_length = self.max_length - len(tokens)
        if pad_length > 0:
            pad_token = self.tokenizer.token_to_id(self.pad_token)
            tokens.extend([pad_token] * pad_length)
            labels_mask.extend([True] * pad_length)  # Ignore padding
        
        # Create input and labels
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Apply mask (-100 is ignored by cross_entropy)
        labels_mask = labels_mask[1:]  # Shift for labels
        labels[labels_mask] = -100
        
        return input_ids, labels

def get_sft_dataloader(tokenizer, batch_size=4, max_length=1024):
    dataset = AlpacaDataset(tokenizer, max_length)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "fr_mixed_tokenizer/gabgpt-vocab.json",
        "fr_mixed_tokenizer/gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)
    dataset = AlpacaDataset(tokenizer)
    dataloader = get_sft_dataloader(tokenizer)
    for batch in dataloader:
        print(batch)
        # decode batch as text with special tokens visible
        print(tokenizer.decode(batch[0][0].tolist()[:100], skip_special_tokens=False))
        # print labels as text with special tokens visible
        # need to decode -100 differently
        # decode -100 with a x
        labels = batch[1][0].tolist()
        labels = [l if l != -100 else 0 for l in labels]
        print(f"with labels ----")
        print(tokenizer.decode(labels, skip_special_tokens=False))
        break