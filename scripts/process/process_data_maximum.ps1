# PowerShell script to maximize dataset size by skipping most filters

# Ensure NLTK punkt data is downloaded
python -c "import nltk; nltk.download('punkt')"

# Run process.py with minimal filtering
python prepare_data\process.py `
    --input_file "scraped_data/data.json" `
    --output_dir "scraped_data_max" `
    --min_abstract_length 200 `
    --min_length_ratio 0.05 `
    --max_length_ratio 5.0 `
    --skip_complexity `
    --skip_term_preservation `
    --skip_paragraph_structure `
    --skip_length_guidance `
    --max_token_length 1020

Write-Host "Processing complete with maximum dataset size!" -ForegroundColor Green
Write-Host "Data saved to: scraped_data_max/data_final_1024.json" -ForegroundColor Cyan
