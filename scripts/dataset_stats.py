#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate statistics for the Gutenberg Chapters dataset.
This script analyzes JSONL files in the data/gutenberg directory and
produces statistics about chapter counts, sizes, and word counts.
"""

import os
import json
import glob
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())

def analyze_dataset(data_dir):
    """
    Analyze the Gutenberg dataset and generate statistics.
    
    Args:
        data_dir: Directory containing the JSONL files
        
    Returns:
        Dictionary containing various statistics
    """
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    
    if not file_paths:
        raise ValueError(f"No JSONL files found in {data_dir}")
    
    print(f"Analyzing {len(file_paths)} files...")
    
    # Initialize statistics
    stats = {
        "total_chapters": 0,
        "chapter_lengths": [],
        "word_counts": [],
        "authors": Counter(),
        "books": Counter(),
        "chapters_per_book": defaultdict(int)
    }
    
    # Process each file
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                
                # Extract metadata
                metadata = entry.get("metadata", {})
                author = metadata.get("author", "Unknown")
                title = metadata.get("title", "Unknown")
                book_id = f"{author} - {title}"
                
                # Extract text and calculate statistics
                text = entry.get("text", "")
                chapter_length = len(text)
                word_count = count_words(text)
                
                # Update statistics
                stats["total_chapters"] += 1
                stats["chapter_lengths"].append(chapter_length)
                stats["word_counts"].append(word_count)
                stats["authors"][author] += 1
                stats["books"][book_id] += 1
                stats["chapters_per_book"][book_id] += 1
    
    # Calculate aggregate statistics
    if stats["chapter_lengths"]:
        stats["min_chapter_length"] = min(stats["chapter_lengths"])
        stats["max_chapter_length"] = max(stats["chapter_lengths"])
        stats["avg_chapter_length"] = np.mean(stats["chapter_lengths"])
        stats["median_chapter_length"] = np.median(stats["chapter_lengths"])
        
        stats["min_word_count"] = min(stats["word_counts"])
        stats["max_word_count"] = max(stats["word_counts"])
        stats["avg_word_count"] = np.mean(stats["word_counts"])
        stats["median_word_count"] = np.median(stats["word_counts"])
        stats["total_words"] = sum(stats["word_counts"])
        
        stats["total_books"] = len(stats["books"])
        stats["total_authors"] = len(stats["authors"])
        
        # Get top authors and books
        stats["top_authors"] = stats["authors"].most_common(10)
        stats["top_books"] = stats["books"].most_common(10)
        
        # Calculate chapters per book statistics
        book_chapter_counts = list(stats["chapters_per_book"].values())
        stats["min_chapters_per_book"] = min(book_chapter_counts)
        stats["max_chapters_per_book"] = max(book_chapter_counts)
        stats["avg_chapters_per_book"] = np.mean(book_chapter_counts)
    
    return stats

def format_stats_for_markdown(stats):
    """
    Format the statistics for inclusion in a markdown file.
    
    Args:
        stats: Dictionary of statistics
        
    Returns:
        Formatted markdown string
    """
    markdown = "## Dataset Statistics\n\n"
    
    # General statistics
    markdown += "### General Statistics\n\n"
    markdown += f"- **Total chapters**: {stats['total_chapters']:,}\n"
    markdown += f"- **Total books**: {stats['total_books']:,}\n"
    markdown += f"- **Total authors**: {stats['total_authors']:,}\n"
    markdown += f"- **Total words**: {stats['total_words']:,}\n\n"
    
    # Chapter statistics
    markdown += "### Chapter Statistics\n\n"
    markdown += f"- **Average chapter length**: {stats['avg_chapter_length']:.2f} characters\n"
    markdown += f"- **Median chapter length**: {stats['median_chapter_length']:.2f} characters\n"
    markdown += f"- **Minimum chapter length**: {stats['min_chapter_length']:,} characters\n"
    markdown += f"- **Maximum chapter length**: {stats['max_chapter_length']:,} characters\n\n"
    
    markdown += f"- **Average word count per chapter**: {stats['avg_word_count']:.2f} words\n"
    markdown += f"- **Median word count per chapter**: {stats['median_word_count']:.2f} words\n"
    markdown += f"- **Minimum word count**: {stats['min_word_count']:,} words\n"
    markdown += f"- **Maximum word count**: {stats['max_word_count']:,} words\n\n"
    
    # Book statistics
    markdown += "### Book Statistics\n\n"
    markdown += f"- **Average chapters per book**: {stats['avg_chapters_per_book']:.2f}\n"
    markdown += f"- **Minimum chapters per book**: {stats['min_chapters_per_book']}\n"
    markdown += f"- **Maximum chapters per book**: {stats['max_chapters_per_book']}\n\n"
    
    # Top authors
    markdown += "### Top Authors by Chapter Count\n\n"
    for author, count in stats["top_authors"]:
        markdown += f"- {author}: {count:,} chapters\n"
    markdown += "\n"
    
    # Top books
    markdown += "### Top Books by Chapter Count\n\n"
    for book, count in stats["top_books"]:
        markdown += f"- {book}: {count:,} chapters\n"
    
    return markdown

def save_stats(stats, output_file):
    """
    Save the statistics to a JSON file.
    
    Args:
        stats: Dictionary of statistics
        output_file: Path to save the JSON file
    """
    # Convert numpy types to Python native types for JSON serialization
    for key, value in stats.items():
        if isinstance(value, np.integer):
            stats[key] = int(value)
        elif isinstance(value, np.floating):
            stats[key] = float(value)
    
    # Remove numpy arrays from the stats
    stats_to_save = {k: v for k, v in stats.items() 
                    if not isinstance(v, (list, np.ndarray))}
    
    # Add summary lists without the full data
    stats_to_save["top_authors"] = stats["top_authors"]
    stats_to_save["top_books"] = stats["top_books"]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to {output_file}")

def analyse_anomalies(stats):
    """
    Analyze and print any anomalies in the statistics.

    Args:
        stats: Dictionary of statistics
    """
    # chapters that are too short
    anomalies = []
    for i, chapter_length in enumerate(stats["chapter_lengths"]):
        if chapter_length < 1000:
            anomalies.append((i, chapter_length))

    print(f"Found {len(anomalies)} anomalies:")

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/gutenberg"
    OUTPUT_JSON = "data/gutenberg/dataset_stats.json"
    OUTPUT_MD = "data/gutenberg/dataset_stats.md"
    
    # Analyze dataset
    stats = analyze_dataset(DATA_DIR)
    analyse_anomalies(stats)
    
    # Save statistics to JSON
    save_stats(stats, OUTPUT_JSON)
    
    # Generate and save markdown
    markdown = format_stats_for_markdown(stats)
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"Markdown statistics saved to {OUTPUT_MD}")
    
    # Print key statistics to console
    print("\nKey Statistics:")
    print(f"Total chapters: {stats['total_chapters']:,}")
    print(f"Total books: {stats['total_books']:,}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Average words per chapter: {stats['avg_word_count']:.2f}")
    print(f"Median words per chapter: {stats['median_word_count']:.2f}")