#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gabgpt'))

try:
    from dataloader import IndexedDataLoader, SingleFileDataLoader
    from tokenizers import ByteLevelBPETokenizer
    import torch
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install torch tokenizers")
    sys.exit(1)

def test_indexed_dataloader():
    """Test IndexedDataLoader with Wikipedia shard data"""
    
    # Load tokenizer
    tokenizer_dir = "fr_mixed_tokenizer/"
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]
    
    tokenizer = ByteLevelBPETokenizer(
        tokenizer_dir + "gabgpt-vocab.json",
        tokenizer_dir + "gabgpt-merges.txt"
    )
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize dataloader
    B = 1  # batch size
    T = 1024  # sequence length
    token_dir = "data/tokenized/wikipedia/"
    
    loader = IndexedDataLoader(
        B=B, 
        T=T, 
        split="train", 
        nb_shards=10, 
        token_dir=token_dir
    )
    
    # Get one batch
    x, y = loader.next_batch()
    
    if x is None:
        print("No data available")
        return
    
    print(f"Batch shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Print first sequence tokens
    tokens = x[0].tolist()
    print(f"\nFirst 50 tokens: {tokens[:50]}")
    
    # Decode the sequence
    decoded_text = tokenizer.decode(tokens)
    print(f"\nDecoded text (first 500 chars):\n{decoded_text[:500]}")
    
    print(f"\nIndexed - Current shard: {loader.get_shard_index()}")

def test_single_file_dataloader():
    """Test SingleFileDataLoader with .bin files"""
    
    # Load tokenizer
    tokenizer_dir = "fr_mixed_tokenizer/"
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]
    
    tokenizer = ByteLevelBPETokenizer(
        tokenizer_dir + "gabgpt-vocab.json",
        tokenizer_dir + "gabgpt-merges.txt"
    )
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize SingleFileDataLoader
    data_dir = "tmp/"  # assuming .bin files are here
    block_size = 1024
    batch_size = 1
    device = torch.device('cpu')
    
    loader = SingleFileDataLoader(
        data_dir=data_dir,
        block_size=block_size,
        batch_size=batch_size,
        device_type='cpu',
        device=device
    )
    
    try:
        # Get one batch
        x, y = loader.get_batch('train')
        
        print(f"SingleFile - Batch shape: {x.shape}")
        print(f"SingleFile - Target shape: {y.shape}")
        
        # Print first sequence tokens
        tokens = x[0].tolist()
        print(f"\nSingleFile : {tokens}")
        # check if 0 is in the tokens and print a warning

        
        
        # Test token 0 mapping
        print(f"\nToken 0 decodes to: '{tokenizer.decode([0])}'")
        print(f"<|endoftext|> encodes to: {tokenizer.encode('<|endoftext|>').ids}")
        
        # Decode normally (special tokens hidden)
        decoded_text = tokenizer.decode(tokens)
        print(f"\nDecoded text (special tokens hidden):\n{decoded_text}")

                # Check if token 0 exists in sequence
        if 0 in tokens:
            pos = tokens.index(0)
            print(f"Token 0 found at position {pos}")
            print(f"Tokens around position {pos}: {tokens[max(0,pos-5):pos+6]}")
            print(f"Decoded text around position {pos}: {tokenizer.decode(tokens[max(0, pos-10):pos+10])}")
        else:
            print("\nNo end of text")

    except Exception as e:
        print(f"SingleFileDataLoader test failed: {e}")

def test_tokenizer_special_tokens():
    """Test special token behavior"""
    tokenizer_dir = "fr_mixed_tokenizer/"
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]
    
    tokenizer = ByteLevelBPETokenizer(
        tokenizer_dir + "gabgpt-vocab.json",
        tokenizer_dir + "gabgpt-merges.txt"
    )
    tokenizer.add_special_tokens(special_tokens)
    
    print("Special token mappings:")
    for token in special_tokens:
        ids = tokenizer.encode(token).ids
        decoded = tokenizer.decode(ids)
        print(f"{token} -> {ids} -> '{decoded}'")

if __name__ == "__main__":
    test_tokenizer_special_tokens()
    print("\n" + "="*50 + "\n")
    test_single_file_dataloader()