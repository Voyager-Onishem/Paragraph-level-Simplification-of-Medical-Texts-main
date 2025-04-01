# Define paths
$MODEL_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both/best_model"
$PRETRAINED_MODEL = "facebook/bart-large-cnn"
$CACHE_DIR = "D:\HF_Cache"

# Set cache environment variables
$env:TRANSFORMERS_CACHE = $CACHE_DIR
$env:HF_HOME = $CACHE_DIR
$env:HF_DATASETS_CACHE = $CACHE_DIR

# Run Python script to fix tokenizer
$pythonScript = @"
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
"@

# Save the Python script to a temporary file
$tempFile = New-TemporaryFile
$pythonScript | Out-File -FilePath "$tempFile.py"

# Execute the Python script
Write-Host "Running Python script to fix tokenizer files..." -ForegroundColor Yellow
python "$tempFile.py"

# Remove the temporary file
Remove-Item "$tempFile.py"

Write-Host "`nNow you can run your generation script normally!" -ForegroundColor Green