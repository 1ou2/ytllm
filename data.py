import json
from tqdm import tqdm
import pandas as pd

def read_jsonl(file_path, max_records=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if max_records and i >= max_records:
                break
            data.append(json.loads(line))
    return data

def extract_paragraphs(section, section_path=""):
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
                results.extend(extract_paragraphs(part, new_section_path))
    
    # If this is a paragraph, add it to results
    elif section.get('type') == 'paragraph':
        results.append((section_path, section.get('value', '')))
    
    # If this is a list, process its parts
    elif section.get('type') == 'list' and 'has_parts' in section:
        for item in section['has_parts']:
                results.extend(extract_paragraphs(item, section_path))
    
    # Process any other type that might have parts
    elif 'has_parts' in section:
        for part in section['has_parts']:
            results.extend(extract_paragraphs(part, section_path))
            
    return results

def create_sample_dataset(src_datafile:str, dest_file:str, size:int):
    """Create a sample dataset from the source data file and save it to the destination file.

    Args:
    src_datafile: path to source file
    dest_file: path to destination file
    size: number of samples to include in file
    """
    data = read_jsonl(src_datafile, size)
    
    # Write directly using json module instead of pandas to ensure proper encoding
    with open(dest_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def display_paragraphs():

    # Load data
    data = read_jsonl('data/frwiki_namespace_0_0.jsonl', 1000)
    df = pd.DataFrame(data)

    # Extract paragraphs from articles
    print("\nExtracting paragraphs from articles:")
    for i in range(min(20, len(df))):  # Process first 3 articles as examples
        row = df.iloc[i]
        print(f"\n\n--- Article {i}: {row.get('name', 'No name')} ---")
        print(f"\n{row.get('sections','no sections')}")
        
        # Process sections
        all_paragraphs = []
        if 'sections' in row:
            for section in row['sections']:
                paragraphs = extract_paragraphs(section)
                all_paragraphs.extend(paragraphs)
        
        # Print all extracted paragraphs with their section paths
        print(f"\nFound {len(all_paragraphs)} paragraphs:")
        for idx, (section_path, paragraph) in enumerate(all_paragraphs, 1):
            print(f"\n{idx}. Section: {section_path}")
            print(f"   Text: {paragraph}")

# Example of how to create a new DataFrame with extracted paragraphs
def extract_all_paragraphs_from_df(df):
    """Extract paragraphs from all articles in the DataFrame"""
    all_data = []
    
    for i, row in df.iterrows():
        article_id = i
        title = row.get('name', 'No Title')
        
        # Process sections
        if 'sections' in row:
            for section in row['sections']:
                paragraphs = extract_paragraphs(section)
                
                for section_path, paragraph_text in paragraphs:
                    all_data.append({
                        'article_id': article_id,
                        'title': title,
                        'section_path': section_path,
                        'paragraph_text': paragraph_text
                    })
    
    return pd.DataFrame(all_data)

# Uncomment to create a DataFrame with all paragraphs
# paragraphs_df = extract_all_paragraphs_from_df(df)
# print(f"\nParagraphs DataFrame shape: {paragraphs_df.shape}")
# print(paragraphs_df.head())

if __name__ == "__main__":
    # Uncomment to create a sample dataset
    create_sample_dataset('data/frwiki_namespace_0_0.jsonl', 'data/sample.jsonl', 100)

    # Uncomment to display paragraphs
    #display_paragraphs()