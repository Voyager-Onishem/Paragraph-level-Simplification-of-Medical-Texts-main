#!/usr/bin/env python
"""Evaluate generated plain-language summaries.

Reads the JSON file produced by `modeling/finetune.py --generate` and computes
ROUGE, BLEU, BERTScore, readability (Flesch, SMOG), and medical term preservation.

Usage:
    python evaluate.py \
        --generations_file trained_models/bart-ul-both/best_model/generation/test_generations.json \
        --output_file evaluation_results.json
"""

import argparse
import json
import os
import traceback

import numpy as np

try:
    import nltk
    nltk.download("punkt", quiet=True)
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError:
    sentence_bleu = None
    print("Warning: 'nltk' not installed – BLEU scores skipped.")

try:
    from rouge import Rouge
    _rouge = Rouge()
except ImportError:
    _rouge = None
    print("Warning: 'rouge' not installed – ROUGE scores skipped.")

try:
    from bert_score import BERTScorer
    _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
except ImportError:
    _bert_scorer = None
    print("Warning: 'bert_score' not installed – BERTScore skipped.")

try:
    from textstat import flesch_reading_ease, smog_index
except ImportError:
    flesch_reading_ease = smog_index = None
    print("Warning: 'textstat' not installed – readability scores skipped.")

_nlp = None
try:
    import spacy
    for _model_name in ["en_core_web_sm", "en_core_web_md"]:
        try:
            _nlp = spacy.load(_model_name)
            break
        except OSError:
            pass
except ImportError:
    pass
if _nlp is None:
    print("Warning: no spaCy model loaded – term preservation skipped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_text(text):
    text = text.replace(" [PARA] ", "\n\n").replace("[PARA]", "\n\n")
    return " ".join(text.split())


def get_entities(text):
    if _nlp is None:
        return set()
    return {ent.text.lower() for ent in _nlp(text).ents}


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def evaluate_one(source, summary, reference):
    entry = {}
    src_words = source.split()
    sum_words = summary.split()
    ref_words = reference.split()

    entry["compression_ratio"] = len(sum_words) / len(src_words) if src_words else 0

    if _rouge:
        try:
            entry["rouge"] = _rouge.get_scores(summary or ".", reference or ".")[0]
        except Exception:
            entry["rouge"] = {}

    if sentence_bleu:
        try:
            entry["bleu"] = sentence_bleu(
                [ref_words], sum_words, smoothing_function=SmoothingFunction().method1
            )
        except Exception:
            entry["bleu"] = 0.0

    if _bert_scorer:
        try:
            P, R, F1 = _bert_scorer.score([summary], [reference])
            entry["bert_score"] = {"precision": P.item(), "recall": R.item(), "f1": F1.item()}
        except Exception:
            entry["bert_score"] = {}

    if flesch_reading_ease:
        try:
            entry["readability"] = {
                "flesch_ease": flesch_reading_ease(summary),
                "smog": smog_index(summary),
            }
        except Exception:
            entry["readability"] = {}

    if _nlp:
        src_terms = get_entities(source)
        sum_terms = get_entities(summary)
        entry["term_preservation"] = len(sum_terms & src_terms) / len(src_terms) if src_terms else 1.0

    return entry


def evaluate_all(generations):
    results = []
    for i, item in enumerate(generations):
        if not isinstance(item, dict):
            continue
        if "source" not in item:
            continue
        source = normalize_text(item["source"])
        summary = normalize_text(item.get("generated", item.get("target", "")))
        reference = normalize_text(item.get("target", ""))
        try:
            entry = evaluate_one(source, summary, reference)
            results.append(entry)
            if i < 5 or i % 20 == 0:
                rouge1_f = entry.get("rouge", {}).get("rouge-1", {}).get("f", 0)
                bleu = entry.get("bleu", 0)
                print(f"[{i}]  ROUGE-1 F={rouge1_f:.4f}  BLEU={bleu:.4f}")
        except Exception:
            print(f"Error on item {i}:\n{traceback.format_exc()}")
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(lst):
    return float(np.mean(lst)) if lst else 0.0


def aggregate(results):
    summary = {}
    for key in ("compression_ratio", "bleu", "term_preservation"):
        vals = [r[key] for r in results if key in r]
        summary[key] = _mean(vals)

    if results and "rouge" in results[0]:
        summary["rouge"] = {}
        for rtype in ("rouge-1", "rouge-2", "rouge-l"):
            summary["rouge"][rtype] = {}
            for metric in ("p", "r", "f"):
                vals = [r["rouge"].get(rtype, {}).get(metric, 0) for r in results if "rouge" in r]
                summary["rouge"][rtype][metric] = _mean(vals)

    if results and "bert_score" in results[0]:
        summary["bert_score"] = {}
        for metric in ("precision", "recall", "f1"):
            vals = [r["bert_score"].get(metric, 0) for r in results if "bert_score" in r]
            summary["bert_score"][metric] = _mean(vals)

    if results and "readability" in results[0]:
        summary["readability"] = {}
        for metric in ("flesch_ease", "smog"):
            vals = [r["readability"].get(metric, 0) for r in results if "readability" in r]
            summary["readability"][metric] = _mean(vals)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated medical text simplifications"
    )
    parser.add_argument("--generations_file", required=True,
                        help="JSON file produced by modeling/finetune.py --generate")
    parser.add_argument("--output_file", default="evaluation_results.json",
                        help="Where to save the aggregated metrics (JSON)")
    args = parser.parse_args()

    print(f"Loading generations from {args.generations_file}")
    with open(args.generations_file, encoding="utf-8") as f:
        data = json.load(f)

    # Support both {"generations": [...]} wrapper and plain list
    if isinstance(data, dict) and "generations" in data:
        generations = data["generations"]
    else:
        generations = data

    print(f"Evaluating {len(generations)} examples …")
    results = evaluate_all(generations)
    print(f"\nEvaluated {len(results)} / {len(generations)} examples")

    summary = aggregate(results)

    print("\n=== Evaluation Results ===")
    print(f"BLEU:               {summary.get('bleu', 0):.4f}")
    if "rouge" in summary:
        print(f"ROUGE-1 F1:         {summary['rouge']['rouge-1']['f']:.4f}")
        print(f"ROUGE-2 F1:         {summary['rouge']['rouge-2']['f']:.4f}")
        print(f"ROUGE-L F1:         {summary['rouge']['rouge-l']['f']:.4f}")
    if "bert_score" in summary:
        print(f"BERTScore F1:       {summary['bert_score']['f1']:.4f}")
    if "readability" in summary:
        print(f"Flesch Reading Ease:{summary['readability']['flesch_ease']:.2f}")
        print(f"SMOG Index:         {summary['readability']['smog']:.2f}")
    print(f"Term Preservation:  {summary.get('term_preservation', 0):.4f}")
    print(f"Compression Ratio:  {summary.get('compression_ratio', 0):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
