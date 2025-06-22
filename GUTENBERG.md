# Gutenberg Chapters Dataset

This dataset contains chapters from french books in the Project Gutenberg collection. Each entry in the dataset represents a single chapter from a book.
The books used for this dataset were filtered on :
- Language : French
- Authors : {'Mérimée', 'Zola', 'Musset', 'Leroux', 'Verne', 'Leblanc', 'Diderot', 'Dumas', 'Proust', 'Voltaire', 'Gautier', 'Stendhal', 'France', 'Balzac', 'Montesquieu', 'Maupassant', 'Daudet'}
- Category : Fiction

## Dataset Structure

Each entry in the dataset contains:

- **metadata**: Information about the source book including:
  - `file_name`: Original file name
  - `title`: Book title
  - `author`: Book author
  - `release_date`: Release date of the book
  - `language`: Language of the book
  - `encoding`: Character encoding of the original file

- **chapter_title**: The title of the chapter (e.g., "CHAPITRE I" or Roman numerals)

- **text**: The full text content of the chapter

## Usage

You can load this dataset using the Hugging Face datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("1ou2/french-classic-fiction-chapters")

# Access the first example
example = dataset['train'][0]
print(f"Chapter: {example['chapter_title']}")
print(f"Book: {example['metadata']['title']} by {example['metadata']['author']}")
print(f"Text preview: {example['text'][:200]}...")
```

## Dataset Creation

This dataset was created by:
1. Collecting text files from Project Gutenberg
2. Preprocessing to remove headers and footers. Fix formatting issues (-- converted to —, _ removed, and fix carriage returns)
3. Identifying chapter boundaries using pattern matching
4. Extracting metadata from the original files
5. Saving each chapter as a separate entry in JSONL format

## License

This dataset contains works from Project Gutenberg. Project Gutenberg books are free and in the public domain in the United States. Please check the copyright laws in your country before using this dataset.