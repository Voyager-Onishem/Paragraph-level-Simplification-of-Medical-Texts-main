# Define paths - Using your fresh "modelsi" directory that isn't backed up
$MODEL_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both/best_model"
$OUTPUT_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both/generations"
$PRETRAINED_MODEL = "facebook/bart-large-cnn"
$DATA_DIR = "D:\Para-Level-Summ Data\data\data-1024"
$CACHE_DIR = "D:\HF_Cache"

# Set number of examples to generate
$NUM_TO_GENERATE = 10  # New parameter to control generation count

# Set cache environment variables
$env:TRANSFORMERS_CACHE = $CACHE_DIR
$env:HF_HOME = $CACHE_DIR
$env:HF_DATASETS_CACHE = $CACHE_DIR

# Create output directory if it doesn't exist
Write-Host "Creating output directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

# REMOVED: The model loading and saving section that was overwriting your checkpoint

# Run the generation
Write-Host "`nRunning generation..." -ForegroundColor Cyan

# Generate with the model
python modeling/finetune.py `
--generate `
--model_name $MODEL_DIR `
--data_dir=$DATA_DIR `
--output_dir=$OUTPUT_DIR `
--generate_mode="test" `
--num_beams=4 `
--min_length=150 `
--template_style="detailed" `
--repetition_penalty=1.2 `
--length_penalty=2.0 `
--early_stopping=False `
--temperature=0.9 `
--min_words=150 `
--num_to_generate=$NUM_TO_GENERATE

Write-Host "`nGeneration complete!" -ForegroundColor Green
Write-Host "Output saved to: $OUTPUT_DIR" -ForegroundColor Green