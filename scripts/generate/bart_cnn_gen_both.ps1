# Set environment variables
$MODEL_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both/best_model"
$DATA_DIR = "D:\Para-Level-Summ Data\data\data-1024"
$OUTPUT_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both/generations"
$CACHE_DIR = "D:\HF_Cache"

# Create output directory
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR

# Set cache environment variables
$env:TRANSFORMERS_CACHE = $CACHE_DIR
$env:HF_HOME = $CACHE_DIR
$env:HF_DATASETS_CACHE = $CACHE_DIR

# Create a timestamp for the log file
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LOG_FILE = "$OUTPUT_DIR/generation_$timestamp.log"

# Generate with the trained model
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
--min_words=150 | Tee-Object -FilePath $LOG_FILE