#!/usr/bin/env python
"""Fine-tune or generate with BART for medical text simplification.

Usage – training:
    python modeling/finetune.py --data_dir data/data-1024 --output_dir trained_models/bart-ul-both \
        --unlikelihood_training --unlikelihood_mode both \
        --cochrane_weights_file data/logr_weights/bart_freq_normalized_ids.txt \
        --newsela_weights_file  data/logr_weights/bart_freq_newsela_ids.txt

Usage – generation:
    python modeling/finetune.py --generate \
        --model_name trained_models/bart-ul-both/best_model \
        --data_dir data/data-1024 --output_dir trained_models/bart-ul-both/best_model/generation
"""

import os
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset (Unchanged)
# ---------------------------------------------------------------------------
class SimplificationDataset(Dataset):
    """Loads line-delimited {split}.source / {split}.target files."""
    def __init__(self, tokenizer, data_dir, split, max_source_length=1024, max_target_length=1024):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        with open(os.path.join(data_dir, f"{split}.source"), encoding="utf-8") as f:
            self.inputs = [line.strip() for line in f]
        with open(os.path.join(data_dir, f"{split}.target"), encoding="utf-8") as f:
            self.targets = [line.strip() for line in f]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source_enc = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            self.targets[idx],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss
        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
        }

# ---------------------------------------------------------------------------
# Vanilla Training Orchestration
# ---------------------------------------------------------------------------
def train(args):
    logger.info("Initializing Tokenizer and Model...")
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    logger.info("Loading Datasets...")
    train_dataset = SimplificationDataset(tokenizer, args.data_dir, "train", args.max_source_length, args.max_target_length)
    val_dataset = SimplificationDataset(tokenizer, args.data_dir, "val", args.max_source_length, args.max_target_length)

    # Standard Hugging Face Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,  
        fp16=True,                    # Mixed precision for speed/memory
        logging_steps=50,            
        predict_with_generate=True,   # Required for Seq2Seq tasks
    )

    # Standard Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting Vanilla Fine-Tuning...")
    trainer.train()
   
    logger.info("Training complete. Saving final model...")
    trainer.save_model(Path(args.output_dir) / "final_model")
    tokenizer.save_pretrained(Path(args.output_dir) / "final_model")

# ---------------------------------------------------------------------------
# CLI Definition
# ---------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser(description="Vanilla fine-tuning of BART for text simplification")
    parser.add_argument("--model_name", default="facebook/bart-large-cnn")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
   
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=1024)
   
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
   
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)


