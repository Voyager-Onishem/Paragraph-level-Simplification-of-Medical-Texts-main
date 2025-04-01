# Set environment variables
$OUTPUT_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both"
$DATA_DIR = "D:\Para-Level-Summ Data\data\data-1024"
$CACHE_DIR = "D:\HF_Cache"

# Create directories
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR
New-Item -ItemType Directory -Force -Path $CACHE_DIR

# Set cache environment variables
$env:TRANSFORMERS_CACHE = $CACHE_DIR
$env:HF_HOME = $CACHE_DIR
$env:HF_DATASETS_CACHE = $CACHE_DIR

# Create a timestamp for the log file
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LOG_FILE = "$OUTPUT_DIR/training_$timestamp.log"

# Run the Python script with correct parameters and redirect output to log file
python modeling/finetune.py `
--model_name facebook/bart-large-cnn `
--data_dir=$DATA_DIR `
--output_dir=$OUTPUT_DIR `
--num_epochs=3 `
--learning_rate=3e-5 `
--train_batch_size=1 `
--eval_batch_size=30 `
--max_source_length=1024 `
--max_target_length=1024 `
--unlikelihood_training `
--unlikelihood_mode=both `
--cochrane_weights_file="D:/Para-Level-Summ Data/data/logr_weights/bart_freq_normalized_ids.txt" `
--newsela_weights_file="D:/Para-Level-Summ Data/data/logr_weights/bart_freq_newsela_ids.txt" `
--exclude_tokens="4,6" `
--unlikelihood_alpha=0.05 `
--gradient_accumulation_steps=4 `
--max_target_ratio=0.9 `
--length_penalty=2.0 | Tee-Object -FilePath $LOG_FILE