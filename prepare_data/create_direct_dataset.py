


"""
Script to create a dataset directly from data.json, bypassing extensive filtering.
This is helpful when the normal processing pipeline filters out too many examples.
"""

import json
import os
import random
from pathlib import Path

def create_direct_dataset():
    print("Creating direct dataset with minimal filtering...")
    
    # Create output directory
    output_dir = 'scraped_data_direct'
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, 'data-1024')
    os.makedirs(data_dir, exist_ok=True)
    
    # Load original data
    try:
        with open('scraped_data/data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f'Loaded {len(data)} articles from data.json')
    except Exception as e:
        print(f'Error loading data: {e}')
        return
    
    # Extract usable content
    processed_data = []
    
    for article in data:
        try:
            # Basic validation
            if not article.get('abstract') or (
               (article.get('pls_type') == 'long' and not article.get('pls')) or
               (article.get('pls_type') == 'sectioned' and not article.get('pls'))):
                continue
            
            # Extract abstract text
            abstract_text = ' '.join(section.get('text', '') 
                                     for section in article.get('abstract', []))
            
            # Extract PLS text based on type
            if article.get('pls_type') == 'long':
                pls_text = article.get('pls', '')
            else:  # sectioned
                pls_text = ' '.join(section.get('text', '') 
                                   for section in article.get('pls', []))
            
            # Very minimal filtering - just check length
            if len(abstract_text) < 100 or len(pls_text) < 50:
                continue
            
            # Keep basic info for the dataset
            processed_data.append({
                'doi': article.get('doi', ''),
                'abstract_text': abstract_text,
                'pls_text': pls_text
            })
            
        except Exception as e:
            print(f'Error processing article {article.get("doi", "unknown")}: {e}')
            continue
    
    print(f'Processed {len(processed_data)} usable articles')
    
    # Save the full processed data
    with open(os.path.join(output_dir, 'data_simplified.json'), 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    # Create train/val/test split
    random.seed(42)  # For reproducibility
    random.shuffle(processed_data)
    
    train_size = int(0.8 * len(processed_data))
    val_size = int(0.1 * len(processed_data))
    
    splits = {
        'train': processed_data[:train_size],
        'val': processed_data[train_size:train_size + val_size],
        'test': processed_data[train_size + val_size:]
    }
    
    print(f'Split sizes: Train: {len(splits["train"])}, Val: {len(splits["val"])}, Test: {len(splits["test"])}')
    
    # Format text with paragraph markers
    def format_text(text):
        return text.replace('\n\n', ' [PARA] ').replace('\n', ' ')
    
    # Write to files in BART training format
    for split_name, split_data in splits.items():
        with (open(f'{data_dir}/{split_name}.doi', 'w', encoding='utf-8') as doi_file,
              open(f'{data_dir}/{split_name}.source', 'w', encoding='utf-8') as source_file,
              open(f'{data_dir}/{split_name}.target', 'w', encoding='utf-8') as target_file):
            
            for item in split_data:
                doi_file.write(f"{item['doi']}\n")
                source_file.write(f"{format_text(item['abstract_text'])}\n")
                target_file.write(f"{format_text(item['pls_text'])}\n")
    
    print(f"Created dataset files in {data_dir}")
    print("Use these files for training with:")
    print(f"python modeling/finetune.py --data_dir={data_dir} --output_dir=trained_models/bart-base")

if __name__ == "__main__":
    create_direct_dataset()
