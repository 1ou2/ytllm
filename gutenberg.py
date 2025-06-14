import os
from pathlib import Path
import re
import chardet
import sys
import json

def clean_text(text:str)->str:
    """Returns a cleaned text version of gutenberg raw text
    - remove carriage returns used after column 70 that are used for formatting purpose only
    - convert -- to — (tiret quadratin)
    - remove _ that are used for emphasis"""
    lines = text.split('\n')
    processed_lines = []
    current_paragraph = []

    for line in lines:
        line = line.strip()

        # Skip empty lines but mark paragraph boundaries
        if not line:
            if current_paragraph:
                processed_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            processed_lines.append('')
            continue

        # Format dialogue with proper French typography
        line = format_french_dialogue(line)
        # Remove emphasis underscores
        line = line.replace('_', '')


        # Check if line starts with '—' (a new paragraph boundary)
        if line.startswith('—'):
            # Finish current paragraph if any
            if current_paragraph:
                processed_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            # short dialog
            if len(line) < 40 and (line.endswith(('.', '!', '?'))):
                processed_lines.append(line)
                continue
            else:
                current_paragraph.append(line)
                continue

        # Handle short lines or sentence endings first
        if len(line) < 40 and (line.endswith(('.', '!', '?')) or is_special_format(line)):
            if current_paragraph:
                current_paragraph.append(line)
                processed_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            else:
                # If there's no current paragraph, add the line as is
                processed_lines.append(line)
            continue
            


        # Check if line should be joined with previous line
        if current_paragraph and should_join_with_previous(current_paragraph[-1], line):
            current_paragraph[-1] = current_paragraph[-1] + ' ' + line
            continue
        else:
            current_paragraph.append(line)

    # Add remaining paragraph
    if current_paragraph:
        processed_lines.append(' '.join(current_paragraph))

    return '\n'.join(processed_lines)



def format_french_dialogue(line):
    """Format dialogue with proper French typography."""
    # initially i wanted to follow french typography rules
    # but i changed my mind !
    nbsp = '\u00A0' # Non-breaking space in Unicode
    nbsp = " " # use normal space
    
    # Replace double dash at the beginning of a line with em dash
    if line.startswith('--'):
        if len(line) > 2 and line[2] == ' ':
            # If followed by a space, replace with em dash + non-breaking space
            line = '—' + nbsp + line[3:]
        else:
            # If followed immediately by a character, add a non-breaking space
            line = '—' + nbsp + line[2:]

    # Some dialogs start with «--
    if line.startswith('«--'):
        line = '—' + nbsp + line[3:]
    
    # Replace double dash within the line
    parts = line.split(' --')
    if len(parts) > 1:
        new_parts = [parts[0]]
        for part in parts[1:]:
            if part and part[0] == ' ':
                # If there's already a space after --, replace with em dash + non-breaking space
                new_parts.append('—' + nbsp + part[1:])
            else:
                # If -- is immediately followed by a character, add a non-breaking space
                new_parts.append('—' + nbsp + part)
        line = ' '.join(new_parts)
    
    return line

def should_join_with_previous(prev_line, current_line):
    """Determine if the current line should be joined with the previous line."""
    # If previous line ends with sentence-ending punctuation, don't join
    if prev_line.endswith(('.', '!', '?')):
        return False
    
    # If previous line ends with a semicolon or colon, it might be the end of a thought
    # but we'll still join if the current line starts with lowercase
    if prev_line.endswith((';', ':')) and current_line and current_line[0].isupper():
        return False
    
    # Otherwise, join the lines
    return True

def is_special_format(line):
    """Check if the line has special formatting that should be preserved."""
    if line.lower().startswith('chapitre'):
        return True
    if re.match(r"^[IVXLC]+\.?$", line.strip()):
        return True
    
    return False



def preprocess():
    # Define files to check
    files_to_check = list(Path("data/raw/gutenberg").glob("*.txt"))
    #files_to_check = list(Path("data/test").glob("*.txt"))
    preprocessed_dir = Path("data/preprocessed/gutenberg")
    # create preprocessed dir if it doesn't exist
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    for file_path in files_to_check:
        lines = preprocess_file(file_path)
        text = '\n'.join(lines)
        # write all lines after startline to file
        # get basename of file and write to "preprocessed" dir
        basename = file_path.name
        preprocessed_path = Path(preprocessed_dir) / basename
        with open(preprocessed_path, 'w', encoding='utf-8') as f:
            f.write(text)    

def preprocess_file(file_path):
    """ Preprocess a file by cleaning header footer, removing formating issues.
    Args :
    - file_path : path to input gutenberg text file
    Returns : a list of lines
    """
    print(f"Preprocessing {file_path}...")
    startline = 0
    endline = -1
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        text = clean_text(raw_text)
        lines = text.split('\n')
        # only keep lines that are not empty
        lines = [line for line in lines if line.strip() != ""]

        # looking for line : *** START OF THE PROJECT GUTENBERG EBOOK
        for i, line in enumerate(lines):
            if line.startswith("***"):
                startline = i +1
                break

        if startline != 0:
            # after what should be the start line we still have other comments in the subsequent lines
            headers = ["produced", "distributed", "proofreading","etext","file","by","http","is","mobipocket"
            "online","available","e-text","the", "Bibliothèque",
            "from","(http","of","at","you","before","whatsoever", "Text", "and the", "we",
            "this", "is", "made","encoded", "note:"]
            for i, line in enumerate(lines[startline:]):
                if line.strip() == "":
                    startline += 1
                else:
                    start_with_header = False
                    # check if line starts with any of the headers
                    for header in headers:
                        if line.lower().startswith(header):
                            startline += 1
                            start_with_header = True
                    # did not find a line starting with a header, nor an empty line
                    # we should be at the start of the book
                    if not start_with_header:
                        break

            # looking for line : *** END OF THE PROJECT GUTENBERG EBOOK
            for i, line in enumerate(lines[startline:]):
                if line.lower().startswith("*** end") or line.startswith("End of "):
                    if endline == -1:
                        endline = i+startline
                    else:
                        endline = min(i+startline, endline)
                    
            # analyse lines backward looking for the word FIN
            # process maximum 10% of the lines
            max_lines = int(len(lines[startline:endline]) * 0.1)
            for i, line in enumerate(reversed(lines[startline:endline][-max_lines:])):
                if line.startswith("FIN"):
                    endline = endline - i -1
                    break
            

        return lines[startline:endline]

def extract_chapters(lines, marker_lines, marker_indexes, filter_func=None):
    """Extract chapters from text based on marker lines and indexes.
    
    Args:
        lines: All lines from the file
        marker_lines: Lines that mark the start of chapters
        marker_indexes: Indexes of the marker lines in the original text
        filter_func: Optional function to filter out unwanted markers
        
    Returns:
        List of chapters (each chapter is a list of lines)
    """
    chapters = []
    chapter_titles = []
    # Filter markers if needed
    if filter_func:
        filtered_lines = []
        filtered_indexes = []
        for i, line in enumerate(marker_lines):
            if filter_func(line):
                filtered_lines.append(line)
                filtered_indexes.append(marker_indexes[i])
        marker_lines = filtered_lines
        marker_indexes = filtered_indexes
    
    # Extract chapters between markers
    if marker_indexes:
        start = marker_indexes[0]
        for index in marker_indexes[1:]:
            # Only append if the next index is more than 2 lines away
            if index - start > 2:
                chapters.append(lines[start+1:index])
                chapter_titles.append(marker_lines[marker_indexes.index(start)])
            start = index
        
        # Handle the last chapter (from last marker to end of file)
        last_index = marker_indexes[-1]
        if len(lines) - last_index > 2:  # Only if there's meaningful content
            chapters.append(lines[last_index+1:])
            chapter_titles.append(marker_lines[marker_indexes.index(last_index)])
            
    return chapters, chapter_titles

def process():
    """Analyse a set of files containing book content coming from the gutenberg project
    Check if we can extract chapters from the book
    """
    processed_dir = Path("data/processed/gutenberg")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    files_to_check = list(Path("data/preprocessed/gutenberg").glob("*.txt"))
    #files_to_check = list(Path("data/test").glob("*.txt"))
    for file_path in files_to_check:
        process_file(file_path, processed_dir)

def process_file(file_path, processed_dir):
    print(f"Analysing {file_path}...")
    lines = []
    with open(file_path, "r",encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        chapter_lines, chapters = process_text(lines)
        if len(chapters) > 0:
            # write chapters to file
            basename = file_path.name
            processed_path = Path(processed_dir) / basename
            with open(processed_path, "w", encoding='utf-8') as f:
                for c in chapters:
                    f.write("\n".join(c) + "\n\n")


def process_text(lines):
    """
    Args:
    - lines : list of text line coming from a book
    Returns : (chapter_lines, chapters)
    chapter_lines : a list of the title of the chapter
    chapters : a list of chapters (text)
    """
    # Search for lines starting with "CHAPITRE"
    chapter_lines = []
    chapter_indexes = []
    for i, line in enumerate(lines):
        if line.lower().startswith("chapitre"):
            # check if the word chapitre appears several times in the entry : it is probably the table of content and should be removed
            if line.lower().count("chapitre") == 1:
                chapter_lines.append(line)
                chapter_indexes.append(i)

    # search for lines starting with a Roman numeral  such as I II III IV
    # Line can only contain a combinaison of these letters plus the dot "."
    roman_lines = []
    roman_indexes = []
    for i, line in enumerate(lines):
        if re.match(r"^[IVXLC]+\.?$", line.strip()):
            roman_lines.append(line)
            roman_indexes.append(i)
        
    if len(chapter_indexes) == 0 and len(roman_indexes) == 0:
        print(f"Could not find a pattern in the file")
        # Could not find a pattern.
        # Try chapters with roman numerals followed by text
        # E.g.: I Followed by text
        for i, line in enumerate(lines):
            if re.match(r"^[IVXLC]+\.?($|\s)", line.strip()):
                roman_lines.append(line)
                roman_indexes.append(i)
    
    # Could not find a pattern
    if len(chapter_indexes) == 0 and len(roman_indexes) == 0:
        print(f"Could not find a pattern in the file")
        return [], []
        
    chapters = []
    chapter_titles = []
    if len(chapter_lines) < 3 and len(roman_lines) > 3:
        # Filter function for roman lines
        def roman_filter(line):
            if ("I " in line and "II " in line and "III " in line):
                return False
            if (" I." in line and " II." in line and " III." in line):
                return False

            return True
            
        chapters, chapter_titles = extract_chapters(lines, roman_lines, roman_indexes, roman_filter)

    elif len(chapter_lines) > 3 and len(roman_lines) < 3:
        # Filter function for chapter lines
        def chapter_filter(line):
            return line.lower().count("chapitre") <= 1
            
        chapters, chapter_titles = extract_chapters(lines, chapter_lines, chapter_indexes, chapter_filter)
    else:
        print(f"Found {len(chapter_lines)} chapters and {len(roman_lines)} roman numerals")
        return [],[]

    
    return chapter_titles, chapters

def get_metadata(file_path):
    """Get metadata from the file name"""
    metadata = {}
    # Extract metadata from file name
    file_name = file_path.name
    metadata["file_name"] = file_name

    with open(file_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # Search for lines starting with "CHAPITRE"
        for line in lines:
            if line.startswith("Title:"):
                metadata["title"] = line.split(":")[1].strip()
            elif line.startswith("Author:"):
                metadata["author"] = line.split(":")[1].strip()
            elif line.startswith("Release Date:"):
                metadata["release_date"] = line.split(":")[1].strip()
            elif line.startswith("Language:"):
                metadata["language"] = line.split(":")[1].strip()
            elif line.startswith("Character set encoding:"):
                metadata["encoding"] = line.split(":")[1].strip()


    return metadata


if __name__ == "__main__":
    
    # Define files to check
    raw_files= list(Path("data/raw/gutenberg").glob("*.txt"))
    data_file = "data/gutenberg.jsonl"

    with open(data_file, "w", encoding='utf-8') as f:
        for file in raw_files:
            metadata = get_metadata(file)
            clean_lines = preprocess_file(file)
            chapter_titles, chapters = process_text(clean_lines)
            assert len(chapter_titles) == len(chapters), f"Number of chapters and titles do not match for {file}"
            if len(chapter_titles) == 0:
                print(f"No chapters found in {file}")
                continue
            gutenberg_entry = {
                "metadata": metadata
            }
            for i, chapter in enumerate(chapters):
                gutenberg_entry["chapter_title"] = chapter_titles[i]
                gutenberg_entry["text"] = "\n".join(chapter)
                f.write(json.dumps(gutenberg_entry, ensure_ascii=False) + "\n")

