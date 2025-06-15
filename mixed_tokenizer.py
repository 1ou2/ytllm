"""
Train a tokenizer on a mix of French datasets (Wikipedia and Oscar).
"""

import os
import random
import json
import glob
import multiprocess as mp
import numpy as np
from tqdm import tqdm
from data import WikipediaFr
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

def sample_news_data(sample_size=100000):
    """
    Sample texts from News dataset.
    
    Args:
        sample_size: Total number of samples to collect
        
    Returns:
        List of text samples
    """
    print("Loading Wikipedia text dataset...")
    news_dataset = load_dataset("1ou2/fr_news_articles", streaming=True)
    train_data = news_dataset["train"]
    train_iter = iter(train_data)
    
    samples = []
    for _ in tqdm(range(sample_size), desc="Sampling News dataset"):
        try:
            sample = next(train_iter)
            samples.append(sample["text"])
        except StopIteration:
            print("Reached end of News dataset")
            break
    
    return samples

def sample_wikipedia_data(sample_size=100000):
    """
    Sample texts from Wikipedia dataset.
    
    Args:
        sample_size: Total number of samples to collect
       
    Returns:
        List of text samples
    """
    print("Loading Wikipedia text dataset...")
    wiki_dataset = load_dataset("1ou2/fr_wiki_paragraphs", streaming=True)
    train_data = wiki_dataset["train"]
    train_iter = iter(train_data)
    
    samples = []
    for _ in tqdm(range(sample_size), desc="Sampling Wikipedia dataset"):
        try:
            sample = next(train_iter)
            samples.append(sample["text"])
        except StopIteration:
            print("Reached end of Wikipedia dataset")
            break
    
    return samples

def sample_book_data(sample_size=100000):
    """
    Sample texts from Gutenberg dataset.

    Args:
        sample_size: Total number of samples to collect

    Returns:
        List of text samples
    """
    print("Loading Gutenberg dataset...")
    gutenberg_dataset = load_dataset("1ou2/french-classic-fiction-chapters",  streaming=True)
    train_data = gutenberg_dataset["train"]
    train_iter = iter(train_data)

    samples = []
    for _ in tqdm(range(sample_size), desc="Sampling Gutenberg dataset"):
        try:
            sample = next(train_iter)
            samples.append(sample["text"])
        except StopIteration:
            print("Reached end of Gutenberg dataset")
            break

    return samples

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
    wiki_ratio=0.74,
    news_ratio=0.25,
    books_ratio=0.01,
    total_samples=500000,
    vocab_size=32768,
    min_frequency=2,
    output_dir="fr_mixed_tokenizer",
    prefix="gabgpt"
):
    """
    Train a tokenizer on a mix of Wikipedia, News, and Books datasets.
    
    Args:
        wiki_ratio: Proportion of samples to take from Wikipedia (default: 0.74)
        news_ratio: Proportion of samples to take from News (default: 0.25)
        books_ratio: Proportion of samples to take from Books (default: 0.01)
        total_samples: Total number of samples to use for training
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        output_dir: Directory to save the tokenizer
        prefix: Prefix for tokenizer files
    """
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>",
                     "<|gab1|>", "<|gab2|>", "<|gab3|>", "<|gab4|>", "<|gab5|>"]
    
    # Validate ratios
    total_ratio = wiki_ratio + news_ratio + books_ratio
    if abs(total_ratio - 1.0) > 0.0001:  # Allow for small floating point errors
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0. Normalizing...")
        wiki_ratio = wiki_ratio / total_ratio
        news_ratio = news_ratio / total_ratio
        books_ratio = books_ratio / total_ratio
    
    # Calculate samples from each source
    wiki_samples = int(total_samples * wiki_ratio)
    news_samples = int(total_samples * news_ratio)
    books_samples = int(total_samples * books_ratio)
    
    # Adjust if rounding caused mismatch
    if wiki_samples + news_samples + books_samples != total_samples:
        # Add any remaining samples to Wikipedia as it's the largest dataset
        wiki_samples = total_samples - news_samples - books_samples
    
    print(f"Sampling strategy: {wiki_samples} Wikipedia samples ({wiki_ratio:.1%}), "
          f"{news_samples} News samples ({news_ratio:.1%}), "
          f"{books_samples} Books samples ({books_ratio:.1%})")
    
    # Sample from Wikipedia
    print("Sampling from Wikipedia...")
    wiki_texts = sample_wikipedia_data(wiki_samples)
    print(f"Collected {len(wiki_texts)} Wikipedia samples")
    
    # Sample from News
    print("Sampling from News...")
    news_texts = sample_news_data(news_samples)
    print(f"Collected {len(news_texts)} News samples")
    
    # Sample from Books
    print("Sampling from Books...")
    books_texts = sample_book_data(books_samples)
    print(f"Collected {len(books_texts)} Books samples")
    
    # Combine and shuffle
    all_texts = wiki_texts + news_texts + books_texts
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
        "news_samples": len(news_texts),
        "books_samples": len(books_texts),
        "total_samples": len(all_texts),
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens
    }
    
    with open(os.path.join(output_dir, f"{prefix}-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return tokenizer


def tokenize_corpus(dataset, tokenizer_dir, output_dir,shard_size=1048577,write_last_shard=False):
    """
    Tokenize the corpus using the trained tokenizer and save the tokens to disk.
    data_set : data to tokenize
    tokenizer_dir = location of vocab.json mand merges.txt
    output_dir
    shard_size : 2^20+1
    """
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]
    print(f"Using custom tokenizer from {tokenizer_dir}")
    vocab_file = os.path.join(tokenizer_dir, "gabgpt-vocab.json")
    merges_file = os.path.join(tokenizer_dir, "gabgpt-merges.txt")
    print(f"vocab file: {vocab_file}")
    print(f"merges file: {merges_file}")
    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        vocab_file,
        merges_file
    )
    eot = tokenizer.token_to_id("<|endoftext|>")
    print(f"eot: {eot}")

    
    os.makedirs(output_dir, exist_ok=True)


    def tokenize(article):
        # all documents start with end of sequence token
        tokens = [eot]
        tokens.extend(tokenizer.encode(article["text"]).ids)

        # convert to a uint16 numpy array - 2 bits per token
        return np.array(tokens, dtype=np.uint16)

    nprocs = max(1, mp.cpu_count() - 1)
    print(f"Using {nprocs} processes")

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # pre-allocate memory for tokens in a shard
        all_tokens_np = np.empty((shard_size,),dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, dataset,chunksize=16):
            if token_count + len(tokens) < shard_size:
                # add tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, desc=f"Tokenizing shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                filename = os.path.join(output_dir, f"shard_{shard_index:06d}.npy")
                # split the document into whatever fits in the shard
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                # save the shard
                np.save(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the shard with the remaining tokens
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        
        # write the last shard
        if write_last_shard and token_count > 0:
            filename = os.path.join(output_dir, f"shard_{shard_index:06d}.npy")
            np.save(filename, all_tokens_np[:token_count])

def use_special_tokens():
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "fr_mixed_tokenizer/gabgpt-vocab.json",
        "fr_mixed_tokenizer/gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)
    txt = "est\n\n\n\r\nest<|endoftext|>"
    print(txt)
    encoded = tokenizer.encode(txt)
    print(encoded.ids)
    decoded_text = tokenizer.decode(encoded.ids)
    print(decoded_text)
    print(tokenizer.encode(decoded_text).ids)
    # print also special tokens
    eot = tokenizer.token_to_id("<|endoftext|>")
    print(f"{eot=}")

def test_training():
    # Example usage with default parameters
    train_mixed_tokenizer(
        wiki_ratio=0.74,  # 74% Wikipedia (higher quality)
        news_ratio=0.25,  # 25% news
        books_ratio=0.01, # 1% books
        total_samples=100000,  # 100K samples total
        vocab_size=32768,
        min_frequency=2,
        output_dir="fr_mixed_tokenizer",
        prefix="gabgpt"
    )
if __name__ == "__main__":
    # test use of special tokens
    #use_special_tokens()

    #test_training()

    # tokenize news corpus
    news = load_dataset("1ou2/fr_news_articles", streaming=True)["train"].shuffle()
    tokenize_corpus(dataset=news, tokenizer_dir="fr_mixed_tokenizer", output_dir="data/tokenized/news", shard_size=1048577)
    print("done news")

    #wikipedia = load_dataset("1ou2/fr_wiki_paragraphs", streaming=True)["train"].shuffle()
    #tokenize_corpus(dataset=wikipedia, tokenizer_dir="fr_mixed_tokenizer", output_dir="data/tokenized/wikipedia", shard_size=1048577)
    #print("done wikipedia")

    #from book import get_chapters
    #zoo = get_chapters("data/texts/zoo.txt")
    #tokenize_corpus(dataset=zoo, tokenizer_dir="fr_mixed_tokenizer", output_dir="data/tokenized/zoo", shard_size=1048577,write_last_shard=True)
    #jeu = get_chapters("data/texts/jeu.txt")
    #tokenize_corpus(dataset=jeu, tokenizer_dir="fr_mixed_tokenizer", output_dir="data/tokenized/jeu", shard_size=1048577,write_last_shard=True)


    #gutenberg = load_dataset("1ou2/french-classic-fiction-chapters", streaming=True)["train"].shuffle()
    #tokenize_corpus(dataset=gutenberg, tokenizer_dir="fr_mixed_tokenizer", output_dir="data/tokenized/gutenberg", shard_size=1048577, write_last_shard=True)
    
