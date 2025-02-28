@echo off
REM Train BART model without unlikelihood training
python modeling\finetune.py ^
--model_name="facebook/bart-large-xsum" ^
--data_dir=data/data-1024 ^
--output_dir=trained_models/bart-no-ul ^
--num_epochs=3 ^
--train_batch_size=1 ^
--eval_batch_size=1 ^
--learning_rate=3e-5 ^
--max_source_length=1024 ^
--max_target_length=1024

REM Additional Options:
REM --gradient_accumulation_steps=4          Accumulate gradients over multiple batches to save memory
REM --save_total_limit=2                     Limit number of checkpoints to save disk space
REM --fp16                                   Use 16-bit floating point precision to reduce memory usage
REM --num_train_epochs=1                     Reduce epochs for faster testing
REM --max_steps=1000                         Limit training to specific number of steps
REM --logging_steps=50                       How often to log training metrics
REM --warmup_steps=500                       Gradual warmup of learning rate
REM --weight_decay=0.01                      L2 regularization to prevent overfitting
REM --val_check_interval=0.25                Run validation every 25% of an epoch
REM --patience=2                             Early stopping after 2 epochs without improvement