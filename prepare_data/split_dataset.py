import json
import random
from os import makedirs
import os

# Check if output directory exists
output_dir = 'scraped_data/data-1024'
if not os.path.exists(output_dir):
    makedirs(output_dir)

# Load data from data_final_1024.json
try:
    with open('scraped_data/data_final_1024.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} articles from data_final_1024.json")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Shuffle data for randomized train/val/test split
random.seed(42)  # For reproducibility
random.shuffle(data)

# Calculate split sizes
num_train = int(0.8 * len(data))
num_val = int(0.1 * len(data))

split_data = {
    'train': data[:num_train],
    'val': data[num_train:num_train+num_val],
    'test': data[num_train+num_val:]
}

print(f"Split sizes: Train: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}")

# Process and write data to files
for split in ['train', 'val', 'test']:
    doi_file = open(f'{output_dir}/{split}.doi', 'w', encoding='utf-8')
    source_file = open(f'{output_dir}/{split}.source', 'w', encoding='utf-8')
    target_file = open(f'{output_dir}/{split}.target', 'w', encoding='utf-8')

    for article in split_data[split]:
        doi = article['doi']
        
        # Use abstract_text and pls_text fields which are strings
        if 'abstract_text' in article and 'pls_text' in article:
            abstract = article['abstract_text']
            pls = article['pls_text']
        else:
            # If abstract_text/pls_text not available, extract from structure
            try:
                # Convert abstract list of sections to text
                abstract_parts = []
                for section in article['abstract']:
                    if isinstance(section, dict) and 'text' in section:
                        abstract_parts.append(section['text'])
                abstract = " ".join(abstract_parts)
                
                # Get PLS text based on type
                if article['pls_type'] == 'long':
                    pls = article['pls']
                else:
                    pls_parts = []
                    for section in article['pls']:
                        if isinstance(section, dict) and 'text' in section:
                            pls_parts.append(section['text'])
                    pls = " ".join(pls_parts)
            except Exception as e:
                print(f"Error processing article {doi}: {e}")
                continue
                
        # Write to files
        doi_file.write(doi + '\n')
        source_file.write(abstract + '\n')
        target_file.write(pls + '\n')

    doi_file.close()
    source_file.close()
    target_file.close()
    print(f"Created {split} files with {len(split_data[split])} examples")

print("Dataset split complete!")

