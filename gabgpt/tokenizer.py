"""
Custom Tokenizer created from french dataset.
"""
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import multiprocess as mp
import os
import numpy as np
from tqdm import tqdm
import lxml.etree as ET
import random

def train():
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>", ]

    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Load the dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.fr")

    # Get 10% of the training split
    texts = dataset["train"].select(range(len(dataset["train"]) // 10))["text"]

    # Train on your dataset while adding special tokens
    tokenizer.train_from_iterator(
        texts, 
        vocab_size=32768, 
        min_frequency=2, 
        special_tokens=special_tokens
    )

    # Save the tokenizer
    # ls french_tokenizer/
    # gabgpt-merges.txt  gabgpt-vocab.json
    tokenizer.save_model("wiki_tokenizer",prefix="gabgpt")

def use_special_tokens():
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "french_tokenizer/gabgpt-vocab.json",
        "french_tokenizer/gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)
    txt = "Bonjour<|endoftext|>"
    encoded = tokenizer.encode(txt)
    print(encoded.ids)
    decoded_text = tokenizer.decode(encoded.ids)
    print(decoded_text)
    print(tokenizer.encode(decoded_text).ids)
    # print also special tokens
    eot = tokenizer.token_to_id("<|endoftext|>")
    print(f"{eot=}")
    


def use_tokenizer(dir):
    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]
    dir = "data/tokenizer/"
    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        f"{dir}" + "gabgpt-vocab.json",
        f"{dir}" + "gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)

    # Test the tokenizer
    test_texts = [
        "Ceci est un exemple de phrase pour tester la tokenization.", 
        "Les réseaux de neurones artificiels sont fascinants !",
        "La programmation est une compétence essentielle pour les développeurs.",
        "L'IA est révolutionnaire dans le domaine de l'intelligence artificielle.",
        "une phrase accentuée à impersonnel et tête avait eu"
    ]

    for text in test_texts:
        output = tokenizer.encode(text)
        print(f"Text: {text}")
        print(f"Tokens: {output.tokens}")
        print(f"IDs: {output.ids}\n")
        print(f"Nb tokens: {len(output.ids)}")

    # Check if special tokens are recognized
    for token in special_tokens:
        encoded = tokenizer.encode(token)
        print(f"Special token '{token}' -> Tokens: {encoded.tokens}")
        print(f"Special token '{token}' -> ID: {encoded.ids}")


    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Method 2: Get the vocabulary size from the tokenizer
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # Optional: You can also inspect some vocabulary items
    print("\nFirst 10 vocabulary items:")
    for token, id in list(vocab.items())[:10]:
        print(f"Token: {token:20} ID: {id}")

    print("\nLast 10 vocabulary items:")
    for token, id in list(vocab.items())[-10:]:
        print(f"Token: {token:20} ID: {id}")

def reddit_dataset():
    file_path="data/raw/spf.xml"

    parser = ET.XMLParser(recover=True)
    #Parses the file
    tree = ET.parse(file_path, parser=parser)
    xroot = tree.getroot()
    dataset = []

    for node in xroot:
        for j in range(len(node.getchildren())):
            text = node.getchildren()[j].text
            dataset.append({"text":text})
    random.shuffle(dataset)
    print(f"Reddit dataset size: {len(dataset)}")
    return dataset

def tokenize_corpus(dataset, tokenizer_dir, output_dir,shard_size=1048577):
    """
    Tokenize the corpus using the trained tokenizer and save the tokens to disk.
    data_set : data to tokenizer
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
        #if token_count > 0:
        #    filename = os.path.join(output_dir, f"shard_{shard_index:06d}.npy")
        #    np.save(filename, all_tokens_np[:token_count])

if __name__ == "__main__":
    #r_dataset = reddit_dataset()
    #tokenize_corpus(r_dataset, "data/tokenizer", "data/tokenized/reddit", shard_size=1048577)

    #w_dataset = load_dataset("wikimedia/wikipedia", "20231101.fr")["train"].shuffle() 
    #tokenize_corpus(w_dataset, "data/tokenizer", "data/tokenized/wikipedia", shard_size=1048577)

    from dataloader import IndexedDataLoader, load_tokens

    red_valid = IndexedDataLoader(1, 64, "valid", "data/tokenized/reddit/valid", process_rank=0, num_processes=1)

    x,y = red_valid.next_batch()
    print(x.shape)
    print(y.shape)
    print(x[0].tolist())
    special_tokens = ["<|endoftext|>", "<|user|>", "<|bot|>", "<|sys|>","<|gab1|>", "<|gab2|>", "<|gab3|>","<|gab4|>", "<|gab5|>"]

    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "french_tokenizer/gabgpt-vocab.json",
        "french_tokenizer/gabgpt-merges.txt"
    )

    # Add special tokens to the loaded tokenizer
    tokenizer.add_special_tokens(special_tokens)
    text = tokenizer.decode(x[0].tolist())
    print(text)