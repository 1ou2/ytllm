import os
from pathlib import Path
import re
import chardet
import sys


# get book files from project gutenberg
monte_cristo = [17989,17990,17991,17992]
non_fiction = [70312,51516,32948,16234,69621,53536,39884,19854,16237,37053]

def detect_encoding(filename):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def test_encoding():
    text = """Les deux frŠres

Le 20 ao–t 1672, la ville de la Haye, si vivante, si"""
    # French special characters (accented 
    # letters, etc.) were entered as DOS upper-  ASCII characters.
    test_string = "frŠres"
    for encoding in ["utf-8",'cp850', 'cp437', 'iso-8859-1',"UTF-8-SIG",'utf-8-sig']:
        try:
            bytes_data = text.encode('latin1')  # Preserve the byte values
            decoded = bytes_data.decode(encoding)
            print(f"{encoding}: {decoded}")
        except:
            print(f"{encoding}: failed")
    
def convert_to_utf8(input_file, output_file):
    # First try CP850 (DOS Latin-1) as it's most likely
    encodings_to_try = ['cp850', 'cp437', 'iso-8859-1','utf-8-sig',"cp1252"]
    
    with open(input_file, 'rb') as file:
        content = file.read()
        
    # Try different encodings
    for encoding in encodings_to_try:
        try:
            decoded_text = content.decode(encoding)
            # If successful, write with UTF-8 encoding
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(decoded_text)
            print(f"Successfully converted using {encoding} to UTF-8")
            return True
        except UnicodeDecodeError:
            continue
    
    print("Could not find correct encoding")
    return False


def download_gutenberg_book(book_id,data_dir):
    gutenberg_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    filename = f"pg{book_id}.txt"
    # check if file already exists
    if os.path.exists(f"{data_dir}/{filename}"):
        print(f"File {filename} already exists")
        return
    os.system(f"wget {gutenberg_url} -P {data_dir}")
   
def get_dumas_id(filename):
    # read lines of dumas.txt files
    with open(filename, "r") as f:
        lines = f.readlines()
    # get the book ids
    book_ids = []
    for line in lines:
        # check if line is an integer
        if line.strip().isdigit():
            book_ids.append(int(line.strip()))
    return sorted(book_ids)

def save_dumas_id(book_ids):
    data_dir = "./data/resources"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open("./data/resources/dumas.txt", "w") as f:
        for book_id in book_ids:
            f.write(f"{book_id}\n")


def preprocess_text2(text):
    # Split the text into lines
    lines = text.split('\n')
    
    # Initialize variables
    processed_lines = []
    current_paragraph = []
    
    for i, line in enumerate(lines):
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
        # remove _ that are used for emphasis 
        line = line.replace('_','')
        
        # Check for dialogue lines (starting with —)
        if line.startswith('—'):
            # Finish the current paragraph if any
            if current_paragraph:
                processed_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            # Add the dialogue line as is
            processed_lines.append(line)
            continue
        
        # For normal text lines, check if they should be joined
        if current_paragraph:
            # Check if this is a continuation of the previous line
            if should_join_with_previous(current_paragraph[-1], line):
                # Join with the previous line
                current_paragraph[-1] = current_paragraph[-1] + ' ' + line
                continue
        
        # Check for intentionally short lines (not continuations)
        # Only treat as separate if it's a complete sentence or special format
        if len(line) < 40 and (line.endswith(('.', '!', '?')) or is_special_format(line)):
            # Finish the current paragraph if any
            if current_paragraph:
                processed_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            # Add the short line as is
            processed_lines.append(line)
        else:
            # Add to the current paragraph
            current_paragraph.append(line)
    
    # Don't forget to add the last paragraph
    if current_paragraph:
        processed_lines.append(' '.join(current_paragraph))
    
    # Join all processed lines
    return '\n'.join(processed_lines)

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
            # Add the dialogue line as is (treated like normal text)
            #processed_lines.append(line)
            #continue

        # For non-dialogue lines, check if they should be joined
        if current_paragraph:
            if should_join_with_previous(current_paragraph[-1], line):
                current_paragraph[-1] = current_paragraph[-1] + ' ' + line
                #continue

        # Handle short lines or sentence endings
        if len(line) < 40 and (line.endswith(('.', '!', '?')) or is_special_format(line)):
            if current_paragraph:
                processed_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            else:
                # If there's no current paragraph, add the line as is
                processed_lines.append(line)

        else:
            current_paragraph.append(line)

    # Add remaining paragraph
    if current_paragraph:
        processed_lines.append(' '.join(current_paragraph))

    return '\n'.join(processed_lines)



def format_french_dialogue(line):
    """Format dialogue with proper French typography."""
    # Non-breaking space in Unicode
    nbsp = '\u00A0'
    
    # Replace double dash at the beginning of a line with em dash
    if line.startswith('--'):
        if len(line) > 2 and line[2] == ' ':
            # If followed by a space, replace with em dash + non-breaking space
            line = '—' + nbsp + line[3:]
        else:
            # If followed immediately by a character, add a non-breaking space
            line = '—' + nbsp + line[2:]
    
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
    # Add any special format checks here
    return False  # For now, no special formats other than those already handled








def preprocess():
    # Define files to check
    files_to_check = list(Path("data/raw/gutenberg").glob("*.txt"))
    preprocessed_dir = Path("data/preprocessed/gutenberg")
    # create preprocessed dir if it doesn't exist
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    for file_path in files_to_check:
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
                    if line.startswith("***"):
                        endline = i+startline
                        break


            # write all lines after startline to file
            # get basename of file and write to "preprocessed" dir
            basename = file_path.name
            preprocessed_path = Path(preprocessed_dir) / basename
            with open(preprocessed_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines[startline:endline]))            


def test_cleaning():

    # Example usage
    text = """Nous avons dit que ce tilbury, destiné à une personne, en charriait
d'ordinaire douze ou quinze; cela, nous le comprenons bien, demande
une explication. Un vieux proverbe français dit: «Quand il y en a
pour un, il y en a pour deux.» Mais je ne connais aucun proverbe dans
aucune langue qui dise: «Quand il y en a pour un, il y en a pour
quinze.»

Il en est cependant ainsi du corricolo, tant, dans les civilisations
avancées, chaque chose est détournée de sa destination primitive!"""

    #processed_text = preprocess_text(text)
    #print(processed_text)

    sample = """
--Oui. Oh! je l'ai vue le jour même de mon premier retour.

--Vous a-t-elle remis les papiers que Marguerite lui avait laissés pour
vous?
D'abord, et presque toujours, un gros moine est assis au milieu, et
forme le centre de l'agglomération humaine que le corricolo emporte
comme un de ces tourbillons d'âmes que Dante vit suivant un grand.
Maintenant, mettez au dessous l'un de l'autre, moine, paysannes,
maris, conducteurs, lazzaroni, gamins et enfans; additionnez le tout,
ajoutez le nourrisson oublié, et vous aurez votre compte. Total,
quinze personnes.
--Bonjour, dit-il.
-- ok
--Et toi ?
Il répondit: --Je vais bien.
_Ceci est en italique._
Texte court ici.
--Je comprends cela, dis-je à Armand, et je suis tout à vous; avez-vous
vu Julie Duprat?

--Oui. Oh! je l'ai vue le jour même de mon premier retour.

--Vous a-t-elle remis les papiers que Marguerite lui avait laissés pour
vous?

--Les voici.

Armand tira un rouleau de dessous son oreiller, et l'y replaça
immédiatement.
Autre texte court.
"""
    #processed_text = preprocess_text(sample)
    #print(processed_text)
    #sys.exit(0)

    data_dir = "./data/raw/gutenberg/"
    file = "pg2419.txt"
    output_dir = "./data/test/"
    with open(f"{data_dir}{file}", encoding="utf-8") as f:
        raw_text = f.read()
        clean_text = preprocess_text(raw_text)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}{file}",mode="w",encoding="utf-8") as t:
            t.write(clean_text)
    sys.exit(0)


if __name__ == "__main__":
    
    #test_cleaning()

    data_dir = "./data/raw/gutenberg"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    book_ids = get_dumas_id("./data/resources/dumas.txt")
    
    for book_id in book_ids:
        download_gutenberg_book(book_id, data_dir)

    preprocess()