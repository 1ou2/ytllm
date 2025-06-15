"""Book data processor"""
import os
import json

def convert_epub_to_txt(epub_path, txt_path):
    """Convert epub to txt"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        
        # Read the epub file
        book = epub.read_epub(epub_path)
        
        # Extract text content
        text_content = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                # Extract text and remove HTML tags
                content = soup.get_text()
                text_content.append(content)
        
        # Write to text file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(text_content))
            
        return True
    except Exception as e:
        print(f"Error converting EPUB to text: {e}")
        return False
    
def analyze_special_chars(txt_path):
    """Analyze text file for special characters
    
    Args:
        txt_path: Path to the text file
        
    Returns:
        Dictionary with special characters and their counts
    """
    special_chars = {}
    common_french_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                             'éèêëàâäôöùûüÿçÉÈÊËÀÂÄÔÖÙÛÜŸÇ'
                             ' ,.;:!?\'"-_()[]{}«»…')
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        for char in content:
            if char not in common_french_chars and char != '\n':
                if char in special_chars:
                    special_chars[char] += 1
                else:
                    special_chars[char] = 1
    
    return special_chars

def get_chapters(txt_path, add_eot=False, normalize_spaces=True, normalize_chars=None):
    """Get chapters from txt file in a format compatible with tokenize_corpus
    
    Args:
        txt_path: Path to the text file
        add_eot: Whether to add <|endoftext|> token after each chapter
        normalize_spaces: Whether to normalize non-breaking spaces (\xa0) to regular spaces
        normalize_chars: Dictionary mapping special characters to their replacements
        
    Returns:
        List of dictionaries with "text" field for each chapter
    """
    chapters = []
    chapter_header = "chapitre"
    
    # Default character normalizations for French text
    if normalize_chars is None:
        normalize_chars = {
            '\xa0': ' ' #,    # Non-breaking space
            #'\u2019': "'",  # Right single quotation mark
            #'\u2013': '-',  # En dash
            #'\u2014': '--', # Em dash
            #'\u2026': '...' # Ellipsis
        }
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        chapter_title = ""
        content = ""
        for line in lines:
            # Normalize characters
            if normalize_spaces:
                line = line.replace('\xa0', ' ')
                
            # Apply other character normalizations
            for char, replacement in normalize_chars.items():
                line = line.replace(char, replacement)
                
            # remove empty lines, but keep trailing carriage return if line not empty
            line = line.strip() + "\n" if line.strip() else ""
            if line.lower().startswith(chapter_header):
                if chapter_title:
                    chapter_text = f"{content}"
                    if add_eot:
                        chapter_text += "<|endoftext|>"
                    chapters.append({"text": chapter_text})
                chapter_title = line.strip()
                content = ""
            else:
                content += line
        # Add the last chapter
        if chapter_title:
            chapter_text = f"{content}"
            if add_eot:
                chapter_text += "<|endoftext|>"
            chapters.append({"text": chapter_text})
    if len(chapters) == 0:
        raise ValueError(f"No chapters found in {txt_path}")
    # remove entries with empty text
    chapters = [chapter for chapter in chapters if chapter["text"].strip() != ""]

    if len(chapters) > 1:
        # Remove last chapter from the list as it is probably incomplete
        chapters.pop()
    return chapters
    
if __name__ == "__main__":
    # Example usage
    epub_path = 'data/books/acelj.epub'
    txt_path = 'data/texts/zoo.txt'

    # First, analyze your text to find special characters
    special_chars = analyze_special_chars(txt_path)
    print("Special characters found:", special_chars)

    #convert_epub_to_txt(epub_path, txt_path):
    #
    chapters = get_chapters(txt_path)
    
    # Print the chapters
    for chapter in chapters:
        if chapter["text"].strip() == "":
            continue
        print(f"------\n{chapter["text"]}\n++++++\n")