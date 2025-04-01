# Simple script to create a dataset with minimal filtering

# First, ensure NLTK punkt data is downloaded
Write-Host "Downloading required NLTK data..." -ForegroundColor Yellow
& python -c "import nltk; nltk.download('punkt')"

# Create a temporary Python file for the direct dataset creation
$pythonScript = @'
import json
import os
import random
import sys

# Create output dirs
direct_dir = 'scraped_data_direct'
os.makedirs(direct_dir, exist_ok=True)
data_dir = os.path.join(direct_dir, 'data-1024')
os.makedirs(data_dir, exist_ok=True)

try:
    # Load original data
    print("Loading original data...")
    with open('scraped_data/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} articles')
    
    # Extract useful content with minimal filtering
    print("Extracting content...")
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
            
            # Minimal filtering - only check if both texts exist
            if len(abstract_text.strip()) < 50 or len(pls_text.strip()) < 50:
                continue
                
            # Save
            processed_data.append({
                'doi': article.get('doi', ''),
                'abstract_text': abstract_text,
                'pls_text': pls_text
            })
        except Exception as e:
            print(f'Error processing article: {e}', file=sys.stderr)
            continue
    
    print(f'Processed {len(processed_data)} articles')
    
    # Save combined data
    with open(os.path.join(direct_dir, 'data_simplified.json'), 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    # Split data for training
    random.seed(42)
    random.shuffle(processed_data)
    
    num_train = int(0.8 * len(processed_data))
    num_val = int(0.1 * len(processed_data))
    
    splits = {
        'train': processed_data[:num_train],
        'val': processed_data[num_train:num_train + num_val],
        'test': processed_data[num_train + num_val:]
    }
    
    print(f'Split sizes: Train: {len(splits["train"])}, Val: {len(splits["val"])}, Test: {len(splits["test"])}')
    
    # Create dataset files
    for split in ['train', 'val', 'test']:
        with open(f'{data_dir}/{split}.doi', 'w', encoding='utf-8') as doi_file, \
             open(f'{data_dir}/{split}.source', 'w', encoding='utf-8') as source_file, \
             open(f'{data_dir}/{split}.target', 'w', encoding='utf-8') as target_file:
             
            for item in splits[split]:
                # Format abstracts and PLS for paragraph structure
                abstract = item['abstract_text'].replace('\n\n', ' [PARA] ').replace('\n', ' ')
                pls = item['pls_text'].replace('\n\n', ' [PARA] ').replace('\n', ' ')
                
                doi_file.write(item['doi'] + '\n')
                source_file.write(abstract + '\n')
                target_file.write(pls + '\n')
    
    print('Created dataset files successfully!')
    
except Exception as e:
    print(f'Error creating direct dataset: {e}', file=sys.stderr)
    raise
'@

# Write the Python code to a temporary file
$pythonFile = Join-Path -Path $env:TEMP -ChildPath "create_dataset.py"
$pythonScript | Out-File -FilePath $pythonFile -Encoding utf8

# Run the Python script
Write-Host "Creating simplified dataset..." -ForegroundColor Cyan
& python $pythonFile

# Also run process.py with very relaxed parameters
Write-Host "Creating alternative dataset with process.py..." -ForegroundColor Green
& python prepare_data\process.py `
    --input_file "scraped_data/data.json" `
    --output_dir "scraped_data_relaxed" `
    --min_abstract_length 50 `
    --min_length_ratio 0.05 `
    --max_length_ratio 5.0 `
    --min_complexity_diff 0 `
    --min_term_preservation 0.1 `
    --min_paragraphs 1 `
    --max_paragraphs 20 `
    --skip_complexity `
    --skip_term_preservation `
    --skip_paragraph_structure `
    --skip_length_guidance

Write-Host "Done! Check these directories for datasets:" -ForegroundColor Green
Write-Host " - scraped_data_direct/data-1024 (direct simplified dataset)" -ForegroundColor Yellow
Write-Host " - scraped_data_relaxed (dataset from process.py with minimal filtering)" -ForegroundColor Yellow

# Clean up
Remove-Item -Path $pythonFile
