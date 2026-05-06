#!/usr/bin/env python
"""Train a logistic regression classifier on abstract vs. PLS token distributions.

The model coefficients become token weights for unlikelihood training:
tokens strongly predictive of complex (abstract) text are given negative weights,
which tells the BART decoder to avoid them when generating plain-language summaries.

Usage:
    python modeling/train_logr.py \
        --data_file  scraped_data/data_final_1024.json \
        --model_file data/logr_model/model.joblib \
        --weights_dir data/logr_weights \
        [--cross_validate]
"""

import argparse
import json
import logging
import os
import sys
import time
from random import shuffle

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from transformers import BartTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def get_abstract(article):
    abstract = article.get("abstract", "")
    if isinstance(abstract, str):
        return abstract
    if isinstance(abstract, list):
        return " ".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in abstract
        )
    return str(abstract)


def get_pls(article):
    pls = article.get("pls", "")
    if isinstance(pls, str):
        return pls
    if isinstance(pls, list):
        return " ".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in pls
        )
    return str(pls)


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------

def make_vector(text, tokenizer):
    """Token-count bag-of-words vector over the BART vocabulary."""
    token_ids = tokenizer.encode(text)[1:-1]  # strip BOS/EOS
    vec = np.zeros(tokenizer.vocab_size, dtype=np.int16)
    for tid in token_ids:
        vec[tid] += 1
    return vec


def build_dataset(data_path, tokenizer):
    logger.info("Loading data from %s", data_path)
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d articles", len(data))
    shuffle(data)

    X = np.zeros((2 * len(data), tokenizer.vocab_size), dtype=np.int16)
    y = np.zeros(2 * len(data), dtype=np.int16)
    idx = 0
    errors = 0
    for i, article in enumerate(data):
        if i % 100 == 0:
            logger.info("Vectorising article %d / %d", i, len(data))
        try:
            X[idx] = make_vector(get_abstract(article), tokenizer)
            X[idx + 1] = make_vector(get_pls(article), tokenizer)
            y[idx] = 0       # abstract (complex)
            y[idx + 1] = 1   # PLS (simple)
            idx += 2
        except Exception as e:
            logger.warning("Skipping article %d: %s", i, e)
            errors += 1

    if errors:
        logger.warning("Skipped %d articles due to errors", errors)
    return X[:idx], y[:idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save(args):
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer)
    logger.info("Tokenizer vocab size: %d", tokenizer.vocab_size)

    X, y = build_dataset(args.data_file, tokenizer)
    logger.info("Dataset shape: %s  class counts: 0=%d 1=%d", X.shape, (y == 0).sum(), (y == 1).sum())

    X_norm = normalize(X.astype(np.float32))
    model = LogisticRegression(max_iter=200)
    t0 = time.time()
    model.fit(X_norm, y)
    logger.info("Training done in %.1fs  train accuracy: %.4f", time.time() - t0, model.score(X_norm, y))

    os.makedirs(os.path.dirname(os.path.abspath(args.model_file)), exist_ok=True)
    dump(model, args.model_file)
    logger.info("Saved model to %s", args.model_file)

    # Build and save sorted vocabulary weight files
    vocab = [
        tokenizer.decode([i], clean_up_tokenization_spaces=False)
        for i in range(tokenizer.vocab_size)
    ]
    weights = np.squeeze(model.coef_, axis=0).tolist()
    sorted_weights = sorted(
        [(i, v, w) for i, (v, w) in enumerate(zip(vocab, weights)) if v.strip()],
        key=lambda x: x[2],
    )

    os.makedirs(args.weights_dir, exist_ok=True)
    ids_path = os.path.join(args.weights_dir, "bart_freq_normalized_ids.txt")
    tokens_path = os.path.join(args.weights_dir, "bart_freq_normalized_tokens.txt")
    with open(ids_path, "w", encoding="utf-8") as f:
        for tid, _, w in sorted_weights:
            f.write(f"{tid} {w}\n")
    with open(tokens_path, "w", encoding="utf-8") as f:
        for _, tok, w in sorted_weights:
            f.write(f"{tok} {w}\n")
    logger.info("Saved token weights to %s", args.weights_dir)

    logger.info("Top-10 simplified tokens: %s", [tok for _, tok, _ in sorted_weights[-10:]])
    logger.info("Top-10 technical  tokens: %s", [tok for _, tok, _ in sorted_weights[:10]])


def cross_validate(args, k=5):
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer)
    X, y = build_dataset(args.data_file, tokenizer)
    X_norm = normalize(X.astype(np.float32))
    splitter = StratifiedKFold(n_splits=k, shuffle=True)
    accs = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_norm, y)):
        m = LogisticRegression(max_iter=200)
        m.fit(X_norm[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], m.predict(X_norm[test_idx]))
        logger.info("Fold %d accuracy: %.4f", fold + 1, acc)
        accs.append(acc)
    logger.info("Cross-val mean: %.4f  std: %.4f", np.mean(accs), np.std(accs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train logistic regression to derive token weights for unlikelihood training"
    )
    parser.add_argument("--data_file", required=True,
                        help="Path to data_final_1024.json produced by prepare_data/process.py")
    parser.add_argument("--model_file", required=True,
                        help="Output path for the trained .joblib model")
    parser.add_argument("--weights_dir", required=True,
                        help="Output directory for bart_freq_normalized_ids.txt / tokens.txt")
    parser.add_argument("--tokenizer", default="facebook/bart-large-cnn",
                        help="Tokenizer used for vectorisation (should match the fine-tuned model)")
    parser.add_argument("--cross_validate", action="store_true",
                        help="Also run k-fold cross-validation after full training")
    args = parser.parse_args()

    train_and_save(args)
    if args.cross_validate:
        cross_validate(args)


if __name__ == "__main__":
    main()
