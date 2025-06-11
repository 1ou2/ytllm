import json
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import datasets
import sys

class WikipediaFr:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.read_jsonl(file_path)
        self.df = pd.DataFrame(self.data)

    def get_paragraphs(self):
        paragraphs = []
        for item in self.data:
            paragraphs.extend(self.extract_paragraphs(item))
        return paragraphs

    def read_jsonl(self,file_path, max_records=None):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f)):
                if max_records and i >= max_records:
                    break
                data.append(json.loads(line))
        return data

    def extract_paragraphs(self, section, section_path=""):
        """
        Recursively extract paragraphs from sections and their nested parts.
        
        Args:
            section: A dictionary representing a section or part
            section_path: String representing the hierarchical path of sections
            
        Returns:
            List of tuples containing (section_path, paragraph_text)
        """
        results = []
        
        # If this is a section, update the section path and process its parts
        if section.get('type') == 'section':
            current_section_name = section.get('name', '')
            if section_path:
                new_section_path = f"{section_path} > {current_section_name}"
            else:
                new_section_path = current_section_name
                
            # Process parts if they exist
            if 'has_parts' in section:
                for part in section['has_parts']:
                    results.extend(self.extract_paragraphs(part, new_section_path))
        
        # If this is a paragraph, add it to results
        elif section.get('type') == 'paragraph':
            results.append((section_path, section.get('value', '')))
        
        # If this is a list, process its parts
        elif section.get('type') == 'list' and 'has_parts' in section:
            for item in section['has_parts']:
                    results.extend(self.extract_paragraphs(item, section_path))
        
        # Process any other type that might have parts
        elif 'has_parts' in section:
            for part in section['has_parts']:
                results.extend(self.extract_paragraphs(part, section_path))
                
        return results

    def create_sample_dataset(self,src_datafile:str, dest_file:str, size:int):
        """Create a sample dataset from the source data file and save it to the destination file.

        Args:
        src_datafile: path to source file
        dest_file: path to destination file
        size: number of samples to include in file
        """
        data = self.read_jsonl(src_datafile, size)
        
        # Write directly using json module instead of pandas to ensure proper encoding
        with open(dest_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def display_paragraphs(self, max_articles)-> None:
        """
        max_articles: number of articles to process
        """
        # Extract paragraphs from articles
        print("\nExtracting paragraphs from articles:")
        for i in range(min(max_articles,len(self.df))):  # Process first articles as examples
            row = self.df.iloc[i]
            print(f"\n\n--- Article {i}: {row.get('name', 'No name')} ---")
            print(f"\n{row.get('sections','no sections')}")
            
            # Process sections
            all_paragraphs = []
            if 'sections' in row:
                for section in row['sections']:
                    paragraphs = self.extract_paragraphs(section)
                    all_paragraphs.extend(paragraphs)
            
            # Print all extracted paragraphs with their section paths
            print(f"\nFound {len(all_paragraphs)} paragraphs:")
            for idx, (section_path, paragraph) in enumerate(all_paragraphs, 1):
                print(f"\n{idx}. Section: {section_path}")
                print(f"   Text: {paragraph}")

    def get_all_articles(self)->list:
        """
        Returns a list of all articles in the DataFrame
        """
        all_articles = []
        for _, row in self.df.iterrows():
            all_articles.append(self.get_article_paragraphs(row))
        
        return all_articles


    def get_article_paragraphs(self, article: pd.DataFrame)-> str:
        """
        df : a datframe containing a wikipedia article
        returns : the article content as a string
        """
        all_paragraphs = []
        # Process sections
        if 'sections' in article:
            for section in article['sections']:
                paragraphs = self.extract_paragraphs(section)
                for _, paragraph_text in paragraphs:
                    all_paragraphs.append(paragraph_text)
        return "\n".join(all_paragraphs)

    # Example of how to create a new DataFrame with extracted paragraphs
    def extract_all_paragraphs(self):
        """Extract paragraphs from all articles in the DataFrame"""
        all_data = []
        
        for i, row in self.df.iterrows():
            article_id = i
            title = row.get('name', 'No Title')
            
            # Process sections
            if 'sections' in row:
                for section in row['sections']:
                    paragraphs = self.extract_paragraphs(section)
                    
                    for section_path, paragraph_text in paragraphs:
                        all_data.append({
                            'article_id': article_id,
                            'title': title,
                            'section_path': section_path,
                            'paragraph_text': paragraph_text
                        })
        
        return pd.DataFrame(all_data)
        
    def create_simplified_dataset(self, output_file):
        """Create a simplified dataset with only text content and save to JSONL file.
        
        Args:
            output_file: Path to save the simplified dataset
        """
        total = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing articles"):
                text = self.get_article_paragraphs(row)
                if text.strip():  # Only include non-empty articles
                    total += 1
                    simplified_entry = {"text": text}
                    f.write(json.dumps(simplified_entry, ensure_ascii=False) + '\n')
        print(f"Total articles processed: {total}")

class Mc4Fr:
    def __init__(self):
        print("Loading dataset...")
        datasets.logging.set_verbosity_debug()
        self.dataset = load_dataset("allenai/c4", "fr", streaming=True)
        print("Dataset loaded.")
    
    def analyse(self):
        # Get the train split and create a single iterator
        train_data = self.dataset["train"]
        train_iter = iter(train_data)

        print("Sample from MC4 dataset:")
        for i in range(5):
            print(f"{i} ---")
            sample = next(train_iter)
            print(sample["text"])

class OscarFr:
    def __init__(self):
        print("Loading dataset...")
        datasets.logging.set_verbosity_debug()
        self.dataset = load_dataset("oscar", "unshuffled_deduplicated_fr", streaming=True, trust_remote_code=True)
        print("Dataset loaded.")
    
    def analyse(self,nb_samples=5):
        # Get the train split and create a single iterator
        train_data = self.dataset["train"]
        # Note: Cannot get size of streaming dataset
        print("Streaming dataset - size unknown")
        train_iter = iter(train_data)
        
        print("Sample from OSCAR dataset:")
        for i in range(nb_samples):
            print(f"{i} ---")
            sample = next(train_iter)
            print(sample["text"])

class MyHgDataset:
    def __init__(self):

        # Load the dataset
        self.dataset = load_dataset('1ou2/fr_wiki_paragraphs')

    def analyse(self):
        # Access the train split
        valid_data = self.dataset['validation']
        print(f"Len valid: {len(valid_data)}")

        # Print the first 5 examples
        for i in range(5):
            print(valid_data[i])

if __name__ == "__main__":
    hgd = MyHgDataset()
    hgd.analyse()
    sys.exit(0)

    # Uncomment to create a sample dataset
    #create_sample_dataset('data/frwiki_namespace_0_0.jsonl', 'data/sample.jsonl', 100)

    #oscar = OscarFr()
    #oscar.analyse(20)
    kaggle_prefix = "frwiki_namespace_0_"
    target_prefix = "frwiki_text_0_"
    extension = ".jsonl"
    indexexs = [14,15]
    for index in indexexs:
        print(f"Processing {kaggle_prefix}{index}{extension}")
        wikisample = WikipediaFr(f"data/{kaggle_prefix}{index}{extension}")
        wikisample.create_simplified_dataset(f"data/{target_prefix}{index}{extension}")
   

    sys.exit(0)

    all_articles = wikisample.get_all_articles()
    print(f"\n\nNumber of articles: {len(all_articles)}")
    print(f"\n\nFifth article: {all_articles[5]}")


    """
    data = read_jsonl("data/sample.jsonl")
    df = pd.DataFrame(data)
    # Uncomment to display paragraphs
    display_paragraphs(df,5)

    for i, row in df.iterrows():
        if i >= 5:
            break
        article_id = i
        title = row.get('name', 'No Title')
        paragraphs = get_paragraphs(row)
        print(f"\n\n--- Article {i}: {title} ---")
        print(f"\n{paragraphs}")

    """

    #mc4 = Mc4Fr()
    #mc4.analyse()
    #oscar = OscarFr()
    #oscar.analyse(20)


