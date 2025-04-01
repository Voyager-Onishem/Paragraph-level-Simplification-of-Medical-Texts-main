# Script to run process.py with minimal filtering and debugging information

# First, ensure NLTK punkt data is downloaded
Write-Host "Downloading required NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt')"

# Examine the original data to understand the issue
Write-Host "Analyzing original data..." -ForegroundColor Yellow
python -c "
import json
try:
    with open('scraped_data/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Original data.json contains {len(data)} articles')
    
    # Check structure sample
    if len(data) > 0:
        sample = data[0]
        print(f'Sample keys: {list(sample.keys())}')
        print(f'Abstract sections: {len(sample.get(\"abstract\", []))}')
        print(f'PLS type: {sample.get(\"pls_type\", \"not found\")}')
        
    # Count PLS types
    pls_types = {}
    for article in data:
        pls_type = article.get('pls_type', 'unknown')
        pls_types[pls_type] = pls_types.get(pls_type, 0) + 1
    print(f'PLS types distribution: {pls_types}')
    
except Exception as e:
    print(f'Error analyzing data: {e}')
"

# Create a direct dataset to bypass all filtering issues
Write-Host "Creating direct dataset bypassing most processing..." -ForegroundColor Cyan
python -c "
import json
import os
import random

# Create output dirs
direct_dir = 'scraped_data_direct'
os.makedirs(direct_dir, exist_ok=True)
data_dir = os.path.join(direct_dir, 'data-1024')
os.makedirs(data_dir, exist_ok=True)

try:
    # Load original data
    with open('scraped_data/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} articles')
    
    # Extract useful content with minimal filtering
    processed_data = []
    for article in data:
        try:
            # Basic validation
            if not article.get('abstract') or (article.get('pls_type') == 'long' and not article.get('pls')) or \
               (article.get('pls_type') == 'sectioned' and not article.get('pls')):
                continue
                
            # Extract abstract text
            abstract_text = ' '.join([section.get('text', '') for section in article.get('abstract', [])])
            
            # Extract PLS text
            if article.get('pls_type') == 'long':
                pls_text = article.get('pls', '')
            else:
                pls_text = ' '.join([section.get('text', '') for section in article.get('pls', [])])
            
            # Minimal filtering
            if len(abstract_text) < 50 or len(pls_text) < 50:
                continue
                
            # Save
            processed_data.append({
                'doi': article.get('doi', ''),
                'abstract_text': abstract_text,
                'pls_text': pls_text
            })
        except Exception as e:
            print(f'Error processing article: {e}')
            continue
    
    print(f'Processed {len(processed_data)} articles')
    
    # Save
    with open(os.path.join(direct_dir, 'data_simplified.json'), 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    # Split
    random.seed(42)
    random.shuffle(processed_data)
    
    num_train = int(0.8 * len(processed_data))
    num_val = int(0.1 * len(processed_data))
    
    splits = {
        'train': processed_data[:num_train],
        'val': processed_data[num_train:num_train + num_val],
        'test': processed_data[num_train + num_val:]
    }
    
    print(f\"Split sizes: Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}\")
    
    # Create dataset files
    for split in ['train', 'val', 'test']:
        with open(f'{data_dir}/{split}.doi', 'w', encoding='utf-8') as doi_file, \
             open(f'{data_dir}/{split}.source', 'w', encoding='utf-8') as source_file, \
             open(f'{data_dir}/{split}.target', 'w', encoding='utf-8') as target_file:
             
            for item in splits[split]:
                # Format abstracts and PLS
                abstract = item['abstract_text'].replace('\\n\\n', ' [PARA] ').replace('\\n', ' ')
                pls = item['pls_text'].replace('\\n\\n', ' [PARA] ').replace('\\n', ' ')
                
                doi_file.write(item['doi'] + '\\n')
                source_file.write(abstract + '\\n')
                target_file.write(pls + '\\n')
    
    print('Created direct dataset files')
    
except Exception as e:
    print(f'Error creating direct dataset: {e}')
"

# Create minimal dataset files with process.py
$debug_dir = "scraped_data_debug"
if (-not (Test-Path $debug_dir)) {
    New-Item -ItemType Directory -Path $debug_dir | Out-Null
}

# Run with minimal filtering settings
Write-Host "Processing with minimal filtering..." -ForegroundColor Green
python prepare_data\process.py `
  --input_file "scraped_data/data.json" `
  --output_dir $debug_dir `
  --min_abstract_length 50 `
  --min_length_ratio 0.05 `
  --max_length_ratio 10.0 `
  --min_complexity_diff 0 `
  --min_term_preservation 0.1 `
  --min_paragraphs 1 `
  --max_paragraphs 50 `
  --skip_complexity `
  --skip_term_preservation `
  --skip_paragraph_structure `
  --skip_length_guidance

Write-Host "Done! Check these directories for datasets:" -ForegroundColor Green
Write-Host " - $debug_dir (with minimal filtering)" -ForegroundColor Yellow
Write-Host " - scraped_data_direct (bypassing most processing)" -ForegroundColor Yellow
