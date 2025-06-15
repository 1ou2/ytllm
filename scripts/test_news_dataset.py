#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the news dataset loading without uploading to Hugging Face.
"""

import os
import json

def test_news_dataset():
    """
    Test loading the news dataset and display sample content.
    """
    data_dir = "data/news"
    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    # Check if files exist
    for path, name in [(train_path, "Train"), (valid_path, "Validation"), (test_path, "Test")]:
        if not os.path.exists(path):
            print(f"Error: {name} file not found at {path}")
            return
        
        # Count lines in file
        with open(path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        print(f"{name} set: {line_count} articles")
        
        # Show sample from each file
        with open(path, 'r', encoding='utf-8') as f:
            sample = f.readline().strip()
            print(f"\n{name} sample (first 150 chars):")
            print(f"{sample[:150]}...\n")
            print("-" * 80)

if __name__ == "__main__":
    test_news_dataset()