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
   
def get_book_ids(filename):
    """Returns a list of gutenberg book ids"""
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

def download_gutenberg_books(book_ids, output_dir):
    """Download books from gutenberg
    book_ids : list of book id
    output_dir : path the target output directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for book_id in book_ids:
        download_gutenberg_book(book_id, output_dir)


def download_dumas_books():
    """Download all books from dumas"""
    data_dir = "./data/raw/gutenberg"
    dumas_id = get_book_ids("./data/resources/dumas.txt")
    download_gutenberg_books(dumas_id, data_dir)


def download_hugo_books():
    """Download all books from hugo"""
    data_dir = "./data/raw/hugo"
    hugo_id = get_book_ids("./data/resources/hugo.txt")
    download_gutenberg_books(hugo_id, data_dir)

def download_french_authors_books():
    """Download all books from french authors"""
    data_dir = "./data/raw/gutenberg"
    french_authors_id = get_book_ids("./data/resources/auteurs-fr.txt")
    download_gutenberg_books(french_authors_id, data_dir)

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
    processed_text = clean_text(sample)
    print(processed_text)
    sys.exit(0)

    data_dir = "./data/raw/gutenberg/"
    file = "pg2419.txt"
    output_dir = "./data/test/"
    with open(f"{data_dir}{file}", encoding="utf-8") as f:
        raw_text = f.read()
        text = clean_text(raw_text)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}{file}",mode="w",encoding="utf-8") as t:
            t.write(text)
    sys.exit(0)


if __name__ == "__main__":
    download_french_authors_books()