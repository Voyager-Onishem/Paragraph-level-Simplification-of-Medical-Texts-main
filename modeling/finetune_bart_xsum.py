#!/usr/bin/env python
# Simplified BART fine-tuning script for medical text simplification

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from torch.cuda.amp import autocast, GradScaler

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

def unlikelihood_loss(logits, decoder_input_ids, weight_vector, epsilon=1e-8):
    # Get probabilities with more stable implementation
    probs = torch.softmax(logits, dim=-1)
    # Clip probabilities to avoid extremes
    probs = torch.clamp(probs, epsilon, 1.0 - epsilon)
    neg_probs = 1 - probs
    
    # Calculate log probabilities directly with clipping
    log_neg_probs = torch.log(neg_probs)
    
    # Create attention mask (ignore padding tokens)
    attention_mask = (decoder_input_ids != 1).float().unsqueeze(2)
    log_neg_probs_masked = log_neg_probs * attention_mask
    
    # Apply weight vector to log probabilities
    weight_mask = weight_vector.unsqueeze(0).unsqueeze(0).expand_as(log_neg_probs_masked)
    weighted_probs = log_neg_probs_masked * weight_mask
    
    # Calculate loss - normalize by number of non-pad tokens for better scaling
    non_pad_tokens = attention_mask.sum() + epsilon
    return -torch.sum(weighted_probs) / non_pad_tokens

def focal_unlikelihood_loss(logits, decoder_input_ids, weight_vector, gamma=2.0, epsilon=1e-8):
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, epsilon, 1.0 - epsilon)
    neg_probs = 1 - probs
    
    # Apply focal weighting - more penalty for high probability tokens
    focal_weights = probs.pow(gamma)
    
    # Calculate log probabilities
    log_neg_probs = torch.log(neg_probs)
    
    # Create attention mask (ignore padding tokens)
    attention_mask = (decoder_input_ids != 1).float().unsqueeze(2)
    
    # Apply focal and token weights
    weight_mask = weight_vector.unsqueeze(0).unsqueeze(0).expand_as(log_neg_probs)
    weighted_probs = log_neg_probs * attention_mask * weight_mask * focal_weights
    
    # Normalize by number of tokens
    non_pad_tokens = attention_mask.sum() + epsilon
    return -torch.sum(weighted_probs) / non_pad_tokens

def token_frequency_aware_loss(logits, decoder_input_ids, weight_vector, token_frequencies, epsilon=1e-8):
    """Token frequency-aware unlikelihood loss"""
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, epsilon, 1.0 - epsilon)
    neg_probs = 1 - probs
    
    # Calculate log probabilities
    log_neg_probs = torch.log(neg_probs)
    
    # Create attention mask (ignore padding tokens)
    attention_mask = (decoder_input_ids != 1).float().unsqueeze(2)
    
    # Apply token frequency weights
    frequency_weights = token_frequencies.unsqueeze(0).unsqueeze(0).expand_as(log_neg_probs)
    weighted_probs = log_neg_probs * attention_mask * weight_vector * frequency_weights
    
    # Normalize by number of tokens
    non_pad_tokens = attention_mask.sum() + epsilon
    return -torch.sum(weighted_probs) / non_pad_tokens

def fix_data_files():
    """Fix mismatched data files by ensuring they have the same number of lines."""
    import os
    
    data_dir = 'data\data-1024'
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
    
    # Add this before model initialization
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("CUDA is NOT available. Using CPU.")
        print("Check installation with: python -c \"import torch; print(torch.cuda.is_available())\"")
        device = torch.device("cpu")

    # Make sure all tensors go to the device
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)
    
    # Move model to device
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
    
    # Load token frequencies
    token_frequencies = torch.ones(model.config.vocab_size)  # Placeholder for actual frequencies
    token_frequencies = token_frequencies.to(device)
    
    # Track metrics
    best_val_loss = float('inf')
    all_losses = []
    
    # Training loop
    logger.info("Starting training")
    scaler = GradScaler()
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            with autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True
                )
                
                # Calculate loss
                loss = outputs.loss / args.gradient_accumulation_steps
            
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
                    
                    # Add token frequency-aware loss
                    tf_loss = token_frequency_aware_loss(
                        outputs.logits,
                        decoder_input_ids,
                        weight_vector,
                        token_frequencies
                    )
                    
                    loss = loss + args.unlikelihood_alpha * (ul_loss + tf_loss)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update parameters
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track loss
            train_loss += loss.item() * args.gradient_accumulation_steps
            
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
            best_model_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
        
        # Save checkpoint for this epoch
   
        best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
    
    # Save training metrics
    with open(output_dir / "training_losses.json", "w") as f:
        json.dump(all_losses, f, indent=2)
    
    logger.info("Training complete")

def generate(args):
    """Generate simplified texts from the model"""
    logger.info(f"Generating using model from: {args.model_name}")
    
    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = SimplificationDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        type_path=args.generate_mode,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
    
    # Adjust end_idx based on sample_size if provided
    if args.sample_size is not None:
        args.end_idx = min(args.start_idx + args.sample_size, len(dataset))
        logger.info(f"Sample size set to {args.sample_size}, will generate {args.end_idx - args.start_idx} examples")
    else:
        args.end_idx = min(args.end_idx, len(dataset))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add special token for paragraphs
    special_tokens = {'additional_special_tokens': ['[PARA]']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Setup generation parameters
    gen_kwargs = {
        "max_length": args.max_target_length,
        "min_length": args.min_length,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_beams": args.num_beams,
        "early_stopping": args.early_stopping,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": args.sampling == "nucleus",
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature
    }
    
    def format_generated_text(text):
        """Post-process generated text to restore paragraph structure."""
        # Replace [PARA] tokens with newlines
        text = text.replace(' [PARA] ', '\n\n')
        # Remove any remaining [PARA] that might be malformed
        text = text.replace('[PARA]', '\n\n')
        # Clean up extra whitespace
        text = ' '.join(text.split())
        # Clean up multiple newlines
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        return text
    
    # Remove hardcoded values and use args
    max_target_ratio = args.max_target_ratio

    # Generate texts
    results = []
    end_idx = min(args.end_idx, len(dataset))
    
    logger.info(f"Generating for examples {args.start_idx} to {end_idx}")
    for idx in range(args.start_idx, end_idx):
        example = dataset[idx]
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)
        
        # First, decode the source text BEFORE using it
        source_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Now we can apply templates and prefixes
        if args.generation_prefix:
            # Add prefix to guide generation
            inputs = tokenizer(args.generation_prefix + source_text, 
                              return_tensors="pt", 
                              max_length=args.max_source_length, 
                              truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

        # Then apply template styles
        template_prefixes = {
            "brief": "Summarize this medical text briefly: ",
            "detailed": "Create a simple, clear summary for patients: ",
            "educational": "Explain in plain language: ",
            "none": ""
        }

        if args.template_style != "none":
            prefix = template_prefixes[args.template_style]
            inputs = tokenizer(prefix + source_text, 
                              return_tensors="pt", 
                              max_length=args.max_source_length, 
                              truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

        # Now generate the text
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
            
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Re-get source and reference for comparison (after possible modifications)
        source_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Fix the labels by replacing -100 with pad_token_id before decoding
        labels = example["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        target_text = tokenizer.decode(labels, skip_special_tokens=True)

        # Then add this check after generation (around line 495):
        if args.min_words > 0:
            # Check if generated text is too short
            word_count = len(generated_text.split())
            target_word_count = len(target_text.split())
            max_target_ratio = 1.15  # Maximum multiple of target length

            if word_count < args.min_words:
                # Retry with stricter parameters
                gen_retry_kwargs = gen_kwargs.copy()
                gen_retry_kwargs["min_length"] = args.min_words * 2
                gen_retry_kwargs["length_penalty"] = 3.0
                gen_retry_kwargs["no_repeat_ngram_size"] = 2
                
                with torch.no_grad():
                    retry_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_retry_kwargs
                    )
                
                retry_text = tokenizer.decode(retry_ids[0], skip_special_tokens=True)
                if len(retry_text.split()) > word_count:
                    generated_text = retry_text
            # If too long, truncate at sentence boundary
            elif word_count > max(args.min_words * 2, target_word_count * max_target_ratio):
                # Truncate at sentence boundary
                sentences = generated_text.split('.')
                truncated_text = ""
                current_word_count = 0
                target_word_count = min(target_word_count * max_target_ratio, args.min_words * 1.5)
                
                for sentence in sentences:
                    if current_word_count > target_word_count:
                        break
                    truncated_text += sentence + "."
                    current_word_count += len(sentence.split())
                
                generated_text = truncated_text
        
        # Apply post-processing to generated text
        generated_text = format_generated_text(generated_text)
        
        # Re-get source and reference for comparison
        source_text = format_generated_text(source_text)
        target_text = format_generated_text(target_text)

        results.append({
            "idx": idx,
            "source": source_text,
            "target": target_text,
            "generated": generated_text,
            "source_length": len(source_text.split()),
            "target_length": len(target_text.split()),
            "generated_length": len(generated_text.split())
        })
        
        if (idx - args.start_idx + 1) % 10 == 0:
            logger.info(f"Generated {idx - args.start_idx + 1}/{end_idx - args.start_idx} examples")
    
    # Import our new evaluator
    from evaluation import SimplificationEvaluator
    evaluator = SimplificationEvaluator(use_spacy=True)
    
    # After generating all results
    sources = [r["source"] for r in results]
    generated = [r["generated"] for r in results]
    references = [r["target"] for r in results]
    
    # Perform comprehensive evaluation
    evaluation_results = evaluator.evaluate_batch(sources, generated, references)
    
    # Add individual metrics to each result
    for i, result in enumerate(results):
        result["metrics"] = evaluator.evaluate_pair(
            result["source"], 
            result["generated"],
            result["target"]
        )
    
    # Add overall metrics to results file
    output = {
        "overall_metrics": evaluation_results,
        "generations": results
    }
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.generate_mode}_generations_with_metrics.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary of metrics
    logger.info("Evaluation Results:")
    for metric, value in evaluation_results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"Saved generations with metrics to {output_file}")

def get_parser():
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of updates steps to accumulate before backward")
    
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
    
    # Add to your ArgumentParser section in finetune.py
    parser.add_argument("--generate", action="store_true", help="Whether to generate text")
    parser.add_argument("--generate_mode", type=str, default="test", help="Which dataset split to generate from (train, val, test)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for generation")
    parser.add_argument("--end_idx", type=int, default=125, help="End index for generation")
    parser.add_argument("--sampling", type=str, default="beam", choices=["beam", "nucleus"], help="Decoding method")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    
    # Add to ArgumentParser (around line 550):
    parser.add_argument("--generation_prefix", type=str, 
                       default="Write a detailed plain language summary that includes key findings: ",
                       help="Prefix to guide generation style and length")

    # Add to ArgumentParser (around line 550):
    parser.add_argument("--min_words", type=int, default=0,
                       help="Minimum word count for generated summaries (0 to disable)")

    # Add to ArgumentParser (around line 550):
    parser.add_argument("--template_style", type=str, 
                       choices=["brief", "detailed", "educational", "none"],
                       default="none",
                       help="Template style for generation")

    parser.add_argument('--sample_size', type=int, default=None,
                      help='Number of examples to generate (for quick validation)')

    # Generation parameters
    parser.add_argument("--min_length", type=int, default=50, help="Minimum generation length")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Size of n-grams to not repeat")
    parser.add_argument("--repetition_penalty", type=float, default=1.5, help="Penalty for repetition")
    parser.add_argument("--length_penalty", type=float, default=1.2, help="Length penalty")
    parser.add_argument("--early_stopping", type=bool, default=True, help="Whether to use early stopping")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature for sampling")
    parser.add_argument("--max_target_ratio", type=float, default=1.15, help="Maximum ratio of generated to target length")

    parser.add_argument("--evaluate_human", action="store_true",
                        help="Run human evaluation interface")
    parser.add_argument("--calculate_kappa", action="store_true",
                        help="Calculate Cohen's Kappa between human and expert ratings")
    parser.add_argument("--expert_ratings_file", type=str,
                        help="File containing expert ratings")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Fix data files before processing
    # fix_data_files()
    
    if args.generate:
        generate(args)
    else:
        train(args)
