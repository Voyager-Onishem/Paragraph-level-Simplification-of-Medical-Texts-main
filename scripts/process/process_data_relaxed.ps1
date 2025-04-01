# PowerShell script with relaxed parameters for a larger dataset

# First, ensure NLTK punkt data is downloaded
python -c "import nltk; nltk.download('punkt')"

# Run process.py with relaxed parameters
python prepare_data\process.py `
    --input_file "scraped_data/data.json" `
    --output_dir "scraped_data" `
    --min_abstract_length 300 `
    --min_length_ratio 0.1 `
    --max_length_ratio 3.0 `
    --min_complexity_diff 2.0 `
    --min_term_preservation 0.25 `
    --min_paragraphs 1 `
    --max_paragraphs 10 `
    --min_compression_ratio 0.15 `
    --max_compression_ratio 1.5

Write-Host "Processing complete! Check scraped_data/data_final_1024.json for results" -ForegroundColor Green
