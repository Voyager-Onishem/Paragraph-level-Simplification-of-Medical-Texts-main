# Define paths
$MODEL_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both/best_model"
$PRETRAINED_MODEL = "facebook/bart-large-cnn"
$CACHE_DIR = "D:\HF_Cache"

# Set cache environment variables
$env:TRANSFORMERS_CACHE = $CACHE_DIR
$env:HF_HOME = $CACHE_DIR
$env:HF_DATASETS_CACHE = $CACHE_DIR

# Create a specific Python file with UTF-8 encoding instead of using a temp file
$pythonFile = ".\fix_tokenizer_script.py"

# Write Python script with explicit UTF-8 encoding
@"
from transformers import BartTokenizer, BartForConditionalGeneration
import os

# Load the original tokenizer from pretrained
tokenizer = BartTokenizer.from_pretrained('$PRETRAINED_MODEL')

# Load the model from the saved directory 
try:
    model = BartForConditionalGeneration.from_pretrained('$MODEL_DIR')
    print(f"Successfully loaded model from {os.path.abspath('$MODEL_DIR')}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Save the tokenizer to the model directory
tokenizer.save_pretrained('$MODEL_DIR')
print(f"Tokenizer files saved to {os.path.abspath('$MODEL_DIR')}")

# List the files in the model directory
print("\nFiles in the model directory after fix:")
for f in os.listdir('$MODEL_DIR'):
    print(f'- {f}')
"@ | Out-File -FilePath $pythonFile -Encoding utf8

# Execute the Python script
Write-Host "Running Python script to fix tokenizer files..." -ForegroundColor Yellow
python $pythonFile

# Remove the temporary file
Remove-Item $pythonFile

Write-Host "`nNow you can run your generation script normally!" -ForegroundColor Green