#!/usr/bin/env python
# Simplified BART fine-tuning script for medical text simplification

import os
import argparse
import json
import logging
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SimplificationDataset(Dataset):
    """Dataset for medical text simplification"""
    
    def __init__(self, tokenizer, data_dir, type_path, max_source_length=1024, max_target_length=1024):
        self.tokenizer = tokenizer
        self.source_path = os.path.join(data_dir, f"{type_path}.source")
        self.target_path = os.path.join(data_dir, f"{type_path}.target")
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        self.inputs = []
        self.targets = []
        
        self._load_data()
    
    def _load_data(self):
        """Load the dataset from files"""
        with open(self.source_path, "r", encoding="utf-8") as f:
            self.inputs = [line.strip() for line in f]
        
        with open(self.target_path, "r", encoding="utf-8") as f:
            self.targets = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        source = self.inputs[idx]
        target = self.targets[idx]
        
        # Tokenize inputs
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Replace padding with -100 (ignored by loss)
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels
        }

def load_weight_vector(filename, vocab_size, exclude_tokens=None):
    """Load token weights for unlikelihood training"""
    if exclude_tokens is None:
        exclude_tokens = set()
        
    # Initialize weight vector
    weight_vector = torch.zeros(vocab_size)
    
    # Read weights from file
    weights = []
    with open(filename) as f:
        for line in f:
            if not line.strip():
                continue
            index, weight = line.strip().split()
            index, weight = int(index), float(weight)
            if index not in exclude_tokens and weight < 0:
                weights.append((index, abs(float(weight))))
    
    # Fill weight vector
    for index, weight in weights:
        weight_vector[index] = weight
    
    return weight_vector

def unlikelihood_loss(logits, decoder_input_ids, weight_vector):
    """Calculate unlikelihood loss to penalize complex tokens"""
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)
    neg_probs = 1 - probs
    
    # Add small constant for numerical stability
    neg_probs = neg_probs + 1e-8
    
    # Calculate log probabilities
    log_neg_probs = torch.log(neg_probs)
    
    # Create attention mask (ignore padding tokens)
    attention_mask = (decoder_input_ids != 1).float().unsqueeze(2)
    log_neg_probs_masked = log_neg_probs * attention_mask
    
    # Apply weight vector to log probabilities
    weight_mask = weight_vector.unsqueeze(0).unsqueeze(0).expand_as(log_neg_probs_masked)
    weighted_probs = log_neg_probs_masked * weight_mask
    
    # Calculate loss
    return -torch.sum(weighted_probs)

def fix_data_files():
    """Fix mismatched data files by ensuring they have the same number of lines."""
    import os
    
    data_dir = 'data/data-1024'
    files_to_check = ['train', 'val', 'test']
    extensions = ['.source', '.target', '.doi']
    
    print("Checking data files for consistency...")
    
    for file_prefix in files_to_check:
        # Get lengths of all files
        file_lengths = []
        file_contents = {}
        
        for ext in extensions:
            filepath = os.path.join(data_dir, file_prefix + ext)
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} does not exist!")
                continue
                
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                file_lengths.append((ext, len(lines)))
                file_contents[ext] = lines
        
        # Find minimum length
        min_length = min([length for _, length in file_lengths])
        
        # Truncate all files to minimum length
        for ext in extensions:
            if ext not in file_contents:
                continue
                
            if len(file_contents[ext]) > min_length:
                print(f"Truncating {file_prefix}{ext} from {len(file_contents[ext])} to {min_length} lines")
                
                # Create backup
                backup_path = os.path.join(data_dir, file_prefix + ext + '.bak')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.writelines(file_contents[ext])
                    
                # Write truncated file
                with open(os.path.join(data_dir, file_prefix + ext), 'w', encoding='utf-8') as f:
                    f.writelines(file_contents[ext][:min_length])
    
    print("Data files have been fixed!")

def train(args):
    """Main training function"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Log args
    logger.info(f"Training arguments: {args}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Load datasets
    logger.info("Loading datasets")
    train_dataset = SimplificationDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        type_path="train",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
    
    val_dataset = SimplificationDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        type_path="val",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    # Configure training
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Setup for unlikelihood training
    if args.unlikelihood_training:
        logger.info(f"Using unlikelihood training mode: {args.unlikelihood_mode}")
        exclude_tokens = set(int(i) for i in args.exclude_tokens.split(',') if i)
        
        if args.unlikelihood_mode == "cochrane":
            weight_vector = load_weight_vector(
                args.cochrane_weights_file, 
                model.config.vocab_size, 
                exclude_tokens
            )
        elif args.unlikelihood_mode == "newsela":
            weight_vector = load_weight_vector(
                args.newsela_weights_file, 
                model.config.vocab_size, 
                exclude_tokens
            )
        elif args.unlikelihood_mode == "both":
            weight_vector1 = load_weight_vector(
                args.cochrane_weights_file, 
                model.config.vocab_size, 
                exclude_tokens
            )
            weight_vector2 = load_weight_vector(
                args.newsela_weights_file, 
                model.config.vocab_size, 
                exclude_tokens
            )
            weight_vector = weight_vector1 + weight_vector2
        
        weight_vector = weight_vector.to(device)
    
    # Track metrics
    best_val_loss = float('inf')
    all_losses = []
    
    # Training loop
    logger.info("Starting training")
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True
            )
            
            # Calculate loss
            loss = outputs.loss
            
            # Add unlikelihood loss if enabled
            if args.unlikelihood_training:
                # Shift decoder input ids
                decoder_input_ids = torch.cat([
                    torch.ones_like(batch["labels"][:, :1]) * model.config.decoder_start_token_id,
                    batch["labels"][:, :-1]
                ], dim=-1)
                decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id
                
                # Calculate unlikelihood loss
                ul_loss = unlikelihood_loss(
                    outputs.logits,
                    decoder_input_ids,
                    weight_vector
                )
                loss = loss + args.unlikelihood_alpha * ul_loss
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track loss
            train_loss += loss.item()
            
            # Log progress
            if (step + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1}, Step {step+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True
                )
                
                # Track loss
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}, Validation loss: {avg_val_loss:.4f}")
        
        # Save losses
        all_losses.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            logger.info(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}, saving model")
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
        
        # Save checkpoint for this epoch
        model.save_pretrained(output_dir / f"checkpoint-{epoch+1}")
        tokenizer.save_pretrained(output_dir / f"checkpoint-{epoch+1}")
    
    # Save training metrics
    with open(output_dir / "training_losses.json", "w") as f:
        json.dump(all_losses, f, indent=2)
    
    logger.info("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BART for medical text simplification")
    
    # Basic parameters
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-xsum", help="Model name or path")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_source_length", type=int, default=1024, help="Max source text length")
    parser.add_argument("--max_target_length", type=int, default=1024, help="Max target text length")
    
    # Unlikelihood training parameters
    parser.add_argument("--unlikelihood_training", action="store_true", help="Whether to use unlikelihood training")
    parser.add_argument("--unlikelihood_mode", type=str, choices=["cochrane", "newsela", "both"], 
                        help="Which weights to use for unlikelihood training")
    parser.add_argument("--cochrane_weights_file", type=str, 
                        default="data/logr_weights/bart_freq_normalized_ids.txt", 
                        help="File containing Cochrane weights")
    parser.add_argument("--newsela_weights_file", type=str, 
                        default="data/logr_weights/bart_freq_newsela_ids.txt",
                        help="File containing Newsela weights")
    parser.add_argument("--exclude_tokens", type=str, default="", help="Comma-separated token IDs to exclude")
    parser.add_argument("--unlikelihood_alpha", type=float, default=0.1, 
                        help="Weight for unlikelihood loss")
    
    args = parser.parse_args()
    
    # Run this function before training
    fix_data_files()
    
    train(args)
