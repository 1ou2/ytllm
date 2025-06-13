import sys
import os
from pathlib import Path
from random import sample
import re
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))




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
            start = index
        
        # Handle the last chapter (from last marker to end of file)
        last_index = marker_indexes[-1]
        if len(lines) - last_index > 2:  # Only if there's meaningful content
            chapters.append(lines[last_index+1:])
            
    return chapters, marker_lines, marker_indexes

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

        print(f"Analysing {file_path}...")
        lines = []
        with open(file_path, "r",encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

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
            continue
            
        chapters = []
        if len(chapter_lines) < 3 and len(roman_lines) > 3:
            print(f"Using roman numerals")
            # Filter function for roman lines
            def roman_filter(line):
                if ("I " in line and "II " in line and "III " in line):
                    return False
                if (" I." in line and " II." in line and " III." in line):
                    return False

                return True
                
            chapters, roman_lines, roman_indexes = extract_chapters(lines, roman_lines, roman_indexes, roman_filter)

        elif len(chapter_lines) > 3 and len(roman_lines) < 3:
            print(f"Using chapters")
            # Filter function for chapter lines
            def chapter_filter(line):
                return line.lower().count("chapitre") <= 1
                
            chapters, chapter_lines, chapter_indexes = extract_chapters(lines, chapter_lines, chapter_indexes, chapter_filter)
        else:
            print(f"Found {len(chapter_lines)} chapters and {len(roman_lines)} roman numerals")

        
        if len(chapters) > 0:
            # write chapters to file
            basename = file_path.name
            processed_path = Path(processed_dir) / basename
            with open(processed_path, "w", encoding='utf-8') as f:
                for c in chapters:
                    f.write("\n".join(c) + "\n\n")

        

if __name__ == "__main__":
    process()