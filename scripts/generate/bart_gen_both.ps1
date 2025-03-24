$MODEL_DIR_NAME = "trained_models/bart-ul_both"
$CURRENT_DIR = Get-Location
$DATA_DIR = Join-Path $CURRENT_DIR "data/data-1024"
$MODEL_DIR = Join-Path $CURRENT_DIR $MODEL_DIR_NAME

python -u modeling/finetune.py `
--model_name "facebook/bart-large-xsum" `
--data_dir "$DATA_DIR" `
--num_epochs 1 `
--learning_rate 3e-5 `
--train_batch_size 1 `
--eval_batch_size 1 `
--output_dir "$MODEL_DIR/generation" `
--max_source_length 1024 `
--max_target_length 1024 `
--generate_mode test `
--batch_size 1 `
--start_idx 0 `
--end_idx 125 `
--sampling nucleus `
--top_p 0.9 `
--generate