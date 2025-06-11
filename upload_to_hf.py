#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to upload datasets to Hugging Face.
This script can handle:
1. JSONL files from the French Wikipedia dataset
2. Text files from the news articles dataset
"""

import os
import glob
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from dotenv import load_dotenv

def create_hf_dataset(data_dir, file_pattern="frwiki_text_*.jsonl"):
    """
    Create a Hugging Face dataset from JSONL files using memory-efficient streaming.
    
    Args:
        data_dir: Directory containing the JSONL files
        file_pattern: Pattern to match files
        
    Returns:
        Hugging Face DatasetDict object
    """
    file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    
    if not file_paths:
        raise ValueError(f"No files found matching pattern {file_pattern} in {data_dir}")
    
    print(f"Found {len(file_paths)} files to process")
    
    # Load all files as a single dataset
    dataset = load_dataset('json', data_files=file_paths)
    
    # Split into train and validation (95% train, 5% validation)
    splits = dataset['train'].train_test_split(test_size=0.05)
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': splits['train'],
        'validation': splits['test']
    })
    
    return dataset_dict

def create_news_dataset(data_dir):
    """
    Create a Hugging Face dataset from news text files.
    Each line in the files represents a complete news article.
    
    Args:
        data_dir: Directory containing the news text files (train.txt, valid.txt, test.txt)
        
    Returns:
        Hugging Face DatasetDict object
    """
    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    # Check if files exist
    for path in [train_path, valid_path, test_path]:
        if not os.path.exists(path):
            raise ValueError(f"Required file not found: {path}")
    
    print(f"Found all required news files in {data_dir}")
    
    # Create dataset dictionary with the text files
    data_files = {
        "train": train_path,
        "validation": valid_path,
        "test": test_path
    }
    
    # Load dataset with text format and rename the column to 'article'
    dataset = load_dataset("text", data_files=data_files)
    
    return dataset

def upload_to_huggingface(dataset, dataset_name, token=None):
    """
    Upload the dataset to Hugging Face.
    
    Args:
        dataset: Hugging Face Dataset object
        dataset_name: Name for the dataset on Hugging Face
        token: Hugging Face API token (optional if already logged in)
    """
    if token:
        login(token)
    
    # Push to hub
    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded successfully to {dataset_name}")

if __name__ == "__main__":
    # Configuration
    load_dotenv()
    hg_token = os.getenv("HUGGINGFACE_TOKEN")
    
    # Choose which dataset to process
    dataset_type = "news"  # Change to "wiki" for the French Wikipedia dataset
    upload_to_hub = True  # Set to True when ready to upload
    
    if dataset_type == "wiki":
        # Process French Wikipedia dataset
        DATA_DIR = "data"
        DATASET_NAME = "fr_wiki_paragraphs"
        dataset = create_hf_dataset(DATA_DIR)
    else:
        # Process news dataset
        DATA_DIR = "data/news"
        DATASET_NAME = "fr_news_articles"  # Change this to your desired name
        dataset = create_news_dataset(DATA_DIR)
    
    # Print dataset info
    print("\nDataset statistics:")
    for split in dataset:
        print(f"{split.capitalize()} set: {len(dataset[split])} examples")
    
    # Show sample data from each split
    print("\nSample data:")
    for split in dataset:
        print(f"\n{split.capitalize()} sample:")
        print(dataset[split][0])
    
    # Upload to Hugging Face if enabled
    if upload_to_hub:
        upload_to_huggingface(dataset, DATASET_NAME, token=hg_token)
    else:
        print("\nUpload to Hugging Face is disabled. Set upload_to_hub=True to enable.")
    
    print("Done!")