$MODEL_DIR_NAME = "trained_models/bart-ul_both/best_model"
$CURRENT_DIR = Get-Location
$DATA_DIR = Join-Path $CURRENT_DIR "data/data-1024"
$MODEL_DIR = Join-Path $CURRENT_DIR $MODEL_DIR_NAME

python -u modeling/finetune.py `
--model_name "$MODEL_DIR" `
--data_dir "$DATA_DIR" `
--output_dir "$MODEL_DIR/generation" `
--num_epochs 1 `
--learning_rate 3e-5 `
--train_batch_size 1 `
--eval_batch_size 1 `
--max_source_length 1024 `
--max_target_length 1024 `
--generate_mode test `
--batch_size 1 `
--start_idx 0 `
--end_idx 125 `
--sampling nucleus `
--top_p 0.9 `
--min_words 75 `
--template_style detailed `
--generation_prefix "Write a comprehensive plain language summary of this medical study, covering the background, methods, results, and implications: " `
--generate