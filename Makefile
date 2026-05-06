# Medical Text Simplification – BART unlikelihood training
# =========================================================
# Run `make help` to list all targets.
#
# Variables can be overridden on the command line, e.g.:
#   make train DATA_DIR=my/data-1024 OUTPUT_DIR=my_run

DATA_DIR      ?= data/data-1024
SCRAPED_DIR   ?= scraped_data
MODEL_NAME    ?= facebook/bart-large-cnn
OUTPUT_DIR    ?= trained_models/bart-ul-both
WEIGHTS_DIR   ?= data/logr_weights
LOGR_MODEL    ?= data/logr_model/model.joblib
GENERATIONS   ?= $(OUTPUT_DIR)/best_model/generation
EVAL_OUT      ?= evaluation_results.json

.PHONY: help scrape process split train-logr train generate evaluate

help:
@echo "Available targets:"
@echo "  scrape      Scrape Cochrane reviews → $(SCRAPED_DIR)/data.json"
@echo "  process     Filter & token-truncate → $(SCRAPED_DIR)/data_final_1024.json"
@echo "  split       Create train/val/test splits → $(SCRAPED_DIR)/data-1024/"
@echo "  train-logr  Train logistic regression for unlikelihood token weights"
@echo "  train       Fine-tune BART with unlikelihood training"
@echo "  generate    Generate plain-language summaries from best_model"
@echo "  evaluate    Evaluate generated summaries (ROUGE, BLEU, BERTScore, …)"
@echo ""
@echo "Override variables, e.g.:  make train OUTPUT_DIR=my_run DATA_DIR=data/data-1024"

scrape:
python prepare_data/scrape.py

process:
python prepare_data/process.py \
--input_file $(SCRAPED_DIR)/data.json \
--output_dir $(SCRAPED_DIR)

split:
python prepare_data/split_dataset.py

train-logr:
python modeling/train_logr.py \
--data_file  $(SCRAPED_DIR)/data_final_1024.json \
--model_file $(LOGR_MODEL) \
--weights_dir $(WEIGHTS_DIR)

train:
python modeling/finetune.py \
--model_name $(MODEL_NAME) \
--data_dir   $(DATA_DIR) \
--output_dir $(OUTPUT_DIR) \
--num_epochs 3 \
--train_batch_size 1 \
--eval_batch_size  4 \
--gradient_accumulation_steps 4 \
--unlikelihood_training \
--unlikelihood_mode both \
--cochrane_weights_file $(WEIGHTS_DIR)/bart_freq_normalized_ids.txt \
--newsela_weights_file  $(WEIGHTS_DIR)/bart_freq_newsela_ids.txt \
--unlikelihood_alpha 0.05

generate:
python modeling/finetune.py \
--generate \
--model_name $(OUTPUT_DIR)/best_model \
--data_dir   $(DATA_DIR) \
--output_dir $(GENERATIONS) \
--num_beams 4 \
--length_penalty 2.0 \
--repetition_penalty 1.5

evaluate:
python evaluate.py \
--generations_file $(GENERATIONS)/test_generations.json \
--output_file $(EVAL_OUT)
