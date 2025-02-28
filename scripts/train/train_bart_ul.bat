@echo off
REM Train BART model with both weight sets
python modeling\finetune.py ^
--model_name="facebook/bart-large-xsum" ^
--data_dir=data/data-1024 ^
--output_dir=trained_models/bart-ul_both ^
--num_epochs=3 ^
--train_batch_size=1 ^
--eval_batch_size=1 ^
--learning_rate=3e-5 ^
--max_source_length=1024 ^
--max_target_length=1024 ^
--unlikelihood_training ^
--unlikelihood_mode=both ^
--unlikelihood_alpha=0.1 ^
--cochrane_weights_file=data/logr_weights/bart_freq_normalized_ids.txt ^
--newsela_weights_file=data/logr_weights/bart_freq_newsela_ids.txt