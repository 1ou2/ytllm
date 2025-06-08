"""
Train a tokenizer on a mix of French datasets (Wikipedia and Oscar).
"""
from tokenizers import ByteLevelBPETokenizer
import os
import random
import json
from tqdm import tqdm
from data import WikipediaFr
from datasets import load_dataset
import glob

def sample_wikipedia_data(wiki_files, sample_size=100000, max_per_file=None):
    """
    Sample texts from Wikipedia dataset.
    
    Args:
        wiki_files: List of Wikipedia JSONL files
        sample_size: Total number of samples to collect
        max_per_file: Maximum samples to take from each file (None for no limit)
    
    Returns:
        List of text samples
    """
    all_samples = []
    samples_per_file = max_per_file or (sample_size // len(wiki_files) + 1)
    
    for file_path in tqdm(wiki_files, desc="Sampling Wikipedia files"):
        wiki_data = WikipediaFr(file_path)
        articles = wiki_data.get_all_articles()
        
        # If we have more articles than needed, randomly sample
        if len(articles) > samples_per_file:
            sampled_articles = random.sample(articles, samples_per_file)
        else:
            sampled_articles = articles
            
        all_samples.extend(sampled_articles)
        
        # Break if we've collected enough samples
        if len(all_samples) >= sample_size:
            break
    
    # Final trim to exact sample size
    return all_samples[:sample_size]

def sample_oscar_data(sample_size=100000):
    """
    Sample texts from Oscar dataset in streaming mode.
    
    Args:
        sample_size: Number of samples to collect
    
    Returns:
        List of text samples
    """
    print("Loading Oscar dataset...")
    oscar_dataset = load_dataset("oscar", "unshuffled_deduplicated_fr", streaming=True, trust_remote_code=True)
    train_data = oscar_dataset["train"]
    train_iter = iter(train_data)
    
    samples = []
    for _ in tqdm(range(sample_size), desc="Sampling Oscar dataset"):
        try:
            sample = next(train_iter)
            samples.append(sample["text"])
        except StopIteration:
            print("Reached end of Oscar dataset")
            break
    
    return samples

def train_mixed_tokenizer(
    wiki_ratio=0.7,
    oscar_ratio=0.3,
    total_samples=500000,
    vocab_size=32768,
    min_frequency=2,
    output_dir="french_tokenizer",
    prefix="gabgpt"
):
    """
    Train a tokenizer on a mix of Wikipedia and Oscar datasets.
    
    Args:
        wiki_ratio: Proportion of samples to take from Wikipedia
        oscar_ratio: Proportion of samples to take from Oscar
        total_samples: Total number of samples to use for training
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        output_dir: Directory to save the tokenizer
        prefix: Prefix for tokenizer files
    """
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>",
                     "<|gab1|>", "<|gab2|>", "<|gab3|>", "<|gab4|>", "<|gab5|>"]
    
    # Calculate samples from each source
    wiki_samples = int(total_samples * wiki_ratio)
    oscar_samples = int(total_samples * oscar_ratio)
    
    # Adjust if rounding caused mismatch
    if wiki_samples + oscar_samples != total_samples:
        wiki_samples = total_samples - oscar_samples
    
    print(f"Sampling strategy: {wiki_samples} Wikipedia samples, {oscar_samples} Oscar samples")
    
    # Get Wikipedia files
    wiki_files = glob.glob("data/frwiki_text_0_*.jsonl")
    if not wiki_files:
        wiki_files = glob.glob("data/frwiki_namespace_0_*.jsonl")
    
    if not wiki_files:
        raise FileNotFoundError("No Wikipedia files found in data directory")
    
    print(f"Found {len(wiki_files)} Wikipedia files: {wiki_files}")
    
    # Sample from Wikipedia
    print("Sampling from Wikipedia...")
    wiki_texts = sample_wikipedia_data(wiki_files, wiki_samples)
    print(f"Collected {len(wiki_texts)} Wikipedia samples")
    
    # Sample from Oscar
    print("Sampling from Oscar...")
    oscar_texts = sample_oscar_data(oscar_samples)
    print(f"Collected {len(oscar_texts)} Oscar samples")
    
    # Combine and shuffle
    all_texts = wiki_texts + oscar_texts
    random.shuffle(all_texts)
    print(f"Total combined samples: {len(all_texts)}")
    
    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Train on the combined dataset
    print("Training tokenizer...")
    tokenizer.train_from_iterator(
        all_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )
    
    # Save the tokenizer
    print(f"Saving tokenizer to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir, prefix=prefix)
    print(f"Tokenizer saved in {output_dir}/")
    
    # Save metadata about the training
    metadata = {
        "wiki_samples": len(wiki_texts),
        "oscar_samples": len(oscar_texts),
        "total_samples": len(all_texts),
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens
    }
    
    with open(os.path.join(output_dir, f"{prefix}-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return tokenizer

if __name__ == "__main__":
    # Example usage with default parameters
    train_mixed_tokenizer(
        wiki_ratio=0.7,  # 70% Wikipedia (higher quality)
        oscar_ratio=0.3,  # 30% Oscar
        total_samples=500000,  # 500K samples total
        vocab_size=32768,
        min_frequency=2,
        output_dir="french_tokenizer",
        prefix="gabgpt"
    )