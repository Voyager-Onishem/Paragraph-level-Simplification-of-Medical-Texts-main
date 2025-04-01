# Set environment variables
$OUTPUT_DIR_NAME = "trained_models/bart-ul_both_detailed"
$CURRENT_DIR = $PWD
$DATA_DIR = "D:\Para-Level-Summ Data\data\data-1024"
$OUTPUT_DIR = "$CURRENT_DIR/$OUTPUT_DIR_NAME"

# Create output directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR

# Run the Python script with adjusted parameters for longer outputs
python modeling/finetune.py `
--model_name facebook/bart-large-xsum `
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
--cochrane_weights_file="data/logr_weights/bart_freq_normalized_ids.txt" `
--newsela_weights_file="data/logr_weights/bart_freq_newsela_ids.txt" `
--exclude_tokens="4,6" `
--unlikelihood_alpha=0.05 `
--gradient_accumulation_steps=4 `
--max_target_ratio=0.9 `
--length_penalty=2.0