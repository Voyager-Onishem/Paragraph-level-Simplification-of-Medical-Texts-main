# Model and data parameters
$MODEL_PATH = "trained_models/bart-ul_both/best_model"
$DATA_DIR = "data/data-1024"
$OUTPUT_DIR = "trained_models/bart-ul_both/best_model/generation"

# Generation control parameters
$SAMPLE_SIZE = 5  # Number of examples to generate
$START_IDX = 0
$GENERATE_MODE = "test"

# Generation configuration
$MIN_LENGTH = 20
$NO_REPEAT_NGRAM_SIZE = 3
$REPETITION_PENALTY = 40
$LENGTH_PENALTY = 1.5
$EARLY_STOPPING = $false
$NUM_RETURN_SEQUENCES = 1
$NUM_BEAMS = 10
$SAMPLING = "nucleus"
$TOP_P = 1.0
$TOP_K = 50
$TEMPERATURE = 0.85
$MAX_TARGET_RATIO = 1.15

# Template and style parameters
$TEMPLATE_STYLE = "detailed"
$MIN_WORDS = 50
$GENERATION_PREFIX = "Write a detailed plain language summary that includes key findings: "

# Construct and execute command
$cmd = "python modeling/finetune.py " + `
    "--generate " + `
    "--model_name `"$MODEL_PATH`" " + `
    "--data_dir `"$DATA_DIR`" " + `
    "--output_dir `"$OUTPUT_DIR`" " + `
    "--sample_size $SAMPLE_SIZE " + `
    "--start_idx $START_IDX " + `
    "--generate_mode $GENERATE_MODE " + `
    "--min_length $MIN_LENGTH " + `
    "--no_repeat_ngram_size $NO_REPEAT_NGRAM_SIZE " + `
    "--repetition_penalty $REPETITION_PENALTY " + `
    "--length_penalty $LENGTH_PENALTY " + `
    "--early_stopping $EARLY_STOPPING " + `
    "--num_return_sequences $NUM_RETURN_SEQUENCES " + `
    "--num_beams $NUM_BEAMS " + `
    "--sampling $SAMPLING " + `
    "--top_p $TOP_P " + `
    "--top_k $TOP_K " + `
    "--temperature $TEMPERATURE " + `
    "--max_target_ratio $MAX_TARGET_RATIO " + `
    "--template_style $TEMPLATE_STYLE " + `
    "--min_words $MIN_WORDS " + `
    "--generation_prefix `"$GENERATION_PREFIX`""

Write-Host "Executing command: $cmd"
Invoke-Expression $cmd
