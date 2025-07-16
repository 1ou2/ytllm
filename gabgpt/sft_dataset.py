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
        
        # Special token IDs
        eos_id = self.tokenizer.token_to_id(self.eos_token)
        pad_id = self.tokenizer.token_to_id(self.pad_token)

        input_ids = []
        labels = []

        for turn in conversation:
            if turn['from'] == 'human':
                user_text = f"{self.user_token}{turn['value']}"
                user_tokens = self.tokenizer.encode(user_text).ids
                input_ids.extend(user_tokens)
                labels.extend([-100] * len(user_tokens))  # mask human input
            elif turn['from'] == 'gpt':
                bot_text = f"{self.bot_token}{turn['value']}"
                bot_tokens = self.tokenizer.encode(bot_text).ids
                input_ids.extend(bot_tokens)
                labels.extend(bot_tokens)  # learn from assistant replies

        # Append EOS token
        input_ids.append(eos_id)
        labels.append(eos_id)

        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # Pad if too short
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([pad_id] * pad_len)
            labels.extend([-100] * pad_len)

        # DO NOT shift labels, as the shift is done in the training
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

  

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
        break
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

    input_ids, labels = dataset[0]
    print("Decoded input:")
    print(tokenizer.decode(input_ids.tolist(), skip_special_tokens=False))
    print("\nDecoded labels (replace -100):")
    decoded = [token if token != -100 else tokenizer.token_to_id("<|gab1|>") for token in labels.tolist()]
    print(tokenizer.decode(decoded, skip_special_tokens=False))
