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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)

# Import the shared evaluator; fall back to sibling-module import when the
# script is run directly (python modeling/finetune.py adds modeling/ to sys.path).
try:
    from modeling.evaluation import SimplificationEvaluator
except ImportError:
    from evaluation import SimplificationEvaluator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
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
# Unlikelihood helpers
# ---------------------------------------------------------------------------

def load_weight_vector(filename, vocab_size, exclude_tokens=None):
    """Build a token-weight tensor from a whitespace-separated id/weight file."""
    if exclude_tokens is None:
        exclude_tokens = set()
    weight_vector = torch.zeros(vocab_size)
    with open(filename) as f:
        for line in f:
            if not line.strip():
                continue
            idx, weight = line.strip().split()
            idx, weight = int(idx), float(weight)
            if idx not in exclude_tokens and weight < 0:
                weight_vector[idx] = abs(weight)
    return weight_vector


def unlikelihood_loss(logits, decoder_input_ids, weight_vector, epsilon=1e-8):
    """Penalise tokens that are statistically associated with complex/technical text."""
    probs = torch.clamp(torch.softmax(logits, dim=-1), epsilon, 1.0 - epsilon)
    log_neg_probs = torch.log(1.0 - probs)
    # Mask padding (pad token id == 1 for BART)
    pad_mask = (decoder_input_ids != 1).float().unsqueeze(2)
    weighted = log_neg_probs * pad_mask * weight_vector.unsqueeze(0).unsqueeze(0).expand_as(log_neg_probs)
    return -torch.sum(weighted) / (pad_mask.sum() + epsilon)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.info("CUDA not available – using CPU.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Fine-tune BART on the processed dataset."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training arguments: %s", args)

    device = _get_device()
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name).to(device)

    train_dataset = SimplificationDataset(
        tokenizer, args.data_dir, "train", args.max_source_length, args.max_target_length
    )
    val_dataset = SimplificationDataset(
        tokenizer, args.data_dir, "val", args.max_source_length, args.max_target_length
    )
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * args.num_epochs,
    )

    # Prepare unlikelihood weight vector
    if args.unlikelihood_training:
        logger.info("Unlikelihood mode: %s", args.unlikelihood_mode)
        exclude = {int(t) for t in args.exclude_tokens.split(",") if t}
        if args.unlikelihood_mode == "cochrane":
            weight_vector = load_weight_vector(args.cochrane_weights_file, model.config.vocab_size, exclude)
        elif args.unlikelihood_mode == "newsela":
            weight_vector = load_weight_vector(args.newsela_weights_file, model.config.vocab_size, exclude)
        else:  # both
            weight_vector = (
                load_weight_vector(args.cochrane_weights_file, model.config.vocab_size, exclude)
                + load_weight_vector(args.newsela_weights_file, model.config.vocab_size, exclude)
            )
        weight_vector = weight_vector.to(device)

    best_val_loss = float("inf")
    all_losses = []
    scaler = GradScaler()

    for epoch in range(args.num_epochs):
        # --- Training pass ---
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                outputs = model(**batch, return_dict=True)
                loss = outputs.loss / args.gradient_accumulation_steps
                if args.unlikelihood_training:
                    decoder_input_ids = torch.cat(
                        [
                            torch.ones_like(batch["labels"][:, :1]) * model.config.decoder_start_token_id,
                            batch["labels"][:, :-1],
                        ],
                        dim=-1,
                    )
                    decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id
                    loss = loss + args.unlikelihood_alpha * unlikelihood_loss(
                        outputs.logits, decoder_input_ids, weight_vector
                    )

            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * args.gradient_accumulation_steps
            if (step + 1) % 50 == 0:
                logger.info(
                    "Epoch %d  step %d/%d  loss %.4f",
                    epoch + 1, step + 1, len(train_loader), loss.item(),
                )

        avg_train_loss = train_loss / len(train_loader)
        logger.info("Epoch %d  avg train loss: %.4f", epoch + 1, avg_train_loss)

        # --- Validation pass ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch, return_dict=True).loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logger.info("Epoch %d  val loss: %.4f", epoch + 1, avg_val_loss)

        all_losses.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info("New best val loss %.4f – saving model", best_val_loss)
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")

        # Save per-epoch checkpoint
        model.save_pretrained(output_dir / f"checkpoint-{epoch + 1}")
        tokenizer.save_pretrained(output_dir / f"checkpoint-{epoch + 1}")

    (output_dir / "training_losses.json").write_text(json.dumps(all_losses, indent=2))
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _fmt(text):
    """Restore paragraph structure from [PARA] tokens and clean whitespace."""
    text = text.replace(" [PARA] ", "\n\n").replace("[PARA]", "\n\n")
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def generate(args):
    """Generate plain-language summaries from a fine-tuned checkpoint."""
    device = _get_device()
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name).to(device)
    model.eval()

    # Register the paragraph special token used during training
    tokenizer.add_special_tokens({"additional_special_tokens": ["[PARA]"]})
    model.resize_token_embeddings(len(tokenizer))

    dataset = SimplificationDataset(
        tokenizer, args.data_dir, args.generate_mode, args.max_source_length, args.max_target_length
    )
    end_idx = min(args.start_idx + (args.num_to_generate or len(dataset)), len(dataset))
    logger.info("Generating examples %d – %d", args.start_idx, end_idx)
    os.makedirs(args.output_dir, exist_ok=True)

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
        "temperature": args.temperature,
    }

    results = []
    for idx in range(args.start_idx, end_idx):
        example = dataset[idx]
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(device)

        # Optionally prepend a generation prefix to guide style/length
        if args.generation_prefix:
            source_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            enc = tokenizer(
                args.generation_prefix + source_text,
                return_tensors="pt",
                max_length=args.max_source_length,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        generated_text = _fmt(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        source_text = _fmt(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        labels = example["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id
        target_text = _fmt(tokenizer.decode(labels, skip_special_tokens=True))

        results.append({
            "idx": idx,
            "source": source_text,
            "target": target_text,
            "generated": generated_text,
            "source_length": len(source_text.split()),
            "target_length": len(target_text.split()),
            "generated_length": len(generated_text.split()),
        })

        if (idx - args.start_idx + 1) % 10 == 0:
            logger.info("Generated %d / %d", idx - args.start_idx + 1, end_idx - args.start_idx)

    # Lightweight evaluation using the shared evaluator
    evaluator = SimplificationEvaluator()
    sources = [r["source"] for r in results]
    generated = [r["generated"] for r in results]
    references = [r["target"] for r in results]
    overall = evaluator.evaluate_batch(sources, generated, references)
    for r in results:
        r["metrics"] = evaluator.evaluate_pair(r["source"], r["generated"], r["target"])

    output = {"overall_metrics": overall, "generations": results}
    out_file = os.path.join(args.output_dir, f"{args.generate_mode}_generations.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Evaluation Results:")
    for metric, value in overall.items():
        logger.info("  %s: %.4f", metric, value)
    logger.info("Saved generations to %s", out_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune or generate with BART for medical text simplification"
    )

    # Shared
    parser.add_argument("--model_name", default="facebook/bart-large-cnn",
                        help="Pretrained model name or path to a fine-tuned checkpoint")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing {train,val,test}.{source,target} files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for saved model checkpoints or generated outputs")
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=1024)

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # Unlikelihood training
    parser.add_argument("--unlikelihood_training", action="store_true",
                        help="Enable unlikelihood loss to suppress technical vocabulary")
    parser.add_argument("--unlikelihood_mode", choices=["cochrane", "newsela", "both"],
                        help="Which token-weight file(s) to use")
    parser.add_argument("--cochrane_weights_file",
                        default="data/logr_weights/bart_freq_normalized_ids.txt")
    parser.add_argument("--newsela_weights_file",
                        default="data/logr_weights/bart_freq_newsela_ids.txt")
    parser.add_argument("--exclude_tokens", default="4,6",
                        help="Comma-separated BART token IDs excluded from unlikelihood loss")
    parser.add_argument("--unlikelihood_alpha", type=float, default=0.05,
                        help="Scaling factor for the unlikelihood loss term")

    # Generation
    parser.add_argument("--generate", action="store_true",
                        help="Run generation instead of training")
    parser.add_argument("--generate_mode", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_to_generate", type=int, default=None,
                        help="Number of examples to generate (default: full split)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--min_length", type=int, default=50)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--length_penalty", type=float, default=1.2)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--sampling", default="beam", choices=["beam", "nucleus"],
                        help="Decoding strategy")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--generation_prefix", default="",
                        help="Optional text prepended to each source before generation")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.generate:
        generate(args)
    else:
        train(args)
