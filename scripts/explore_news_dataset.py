# -*- coding: utf-8 -*-

"""
Random Line Fetcher for Large Datasets

This script allows users to randomly fetch and display lines from a large dataset.
An index file is created to keep track of the positions of each line in the dataset, 
allowing for efficient random line retrieval.

Author     : Guillaume Eckendoerffer
Date       : 06-07-23
Repository : https://github.com/Eckendoerffer/TorchTrainerFlow/
"""

import os
import random
import json

# Path configurations
path = os.path.dirname(os.path.abspath(__file__))
path_dataset = os.path.join(path, "valid.txt")
path_index = os.path.join(path, "dataset_news_index.txt")  # Index file

# Flag to determine if an offset of one byte should be applied
shift_one = True

def build_index():
    """
    Constructs an index for the dataset where each line's starting position is stored.
    """
    index = []
    with open(path_dataset, 'r', encoding='utf-8') as f:
        offset = 0
        for line in f:
            index.append(offset)
            offset += len(line.encode('utf-8'))  

    with open(path_index, 'w', encoding="utf8") as f:
        json.dump(index, f)

def get_line(file_path, line_number, index, i):
    """
    Fetches a specific line from the dataset using the provided index.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        if shift_one:
            f.seek(index[line_number] + line_number)
        else:
            f.seek(index[line_number])
        text = f.readline()

        show = f"{i}) {text}\n"       
        show += ' ' + '-'*220 + '\n' 
        return show

# Build index if it doesn't exist
if not os.path.exists(path_index):
    build_index()

# Load the index file
with open(path_index, 'r') as file:
    index = json.load(file)

# Display 10 random lines from the dataset
for i in range(10):
    print(get_line(path_dataset, random.randint(0, len(index)-1), index, i+1)) 











