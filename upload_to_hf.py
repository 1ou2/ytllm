#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to upload French Wikipedia dataset to Hugging Face.
This script reads JSONL files from the data directory and creates a dataset for uploading to Hugging Face.
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
    DATA_DIR = "data"
    DATASET_NAME = "fr_wiki_paragraphs"  # Change this to your desired name
    load_dotenv()
    hg_token = os.getenv("HUGGINGFACE_TOKEN")
    # Create Hugging Face dataset
    dataset = create_hf_dataset(DATA_DIR)
    
    # Print dataset info
    print("\nDataset statistics:")
    print(f"Train set: {len(dataset['train'])} examples")
    print(f"Validation set: {len(dataset['validation'])} examples")
    
    # Upload to Hugging Face
    # Uncomment the line below and add your token to upload
    upload_to_huggingface(dataset, DATASET_NAME, token=hg_token)
    
    print("Done!")