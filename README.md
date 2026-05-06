# Medical Text Simplification with BART

Fine-tuning BART to convert Cochrane Review abstracts into Plain Language Summaries (PLS), using **unlikelihood training** to suppress medical jargon and **paragraph-aware formatting** to produce well-structured output.

## Project overview

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Scrape | `prepare_data/scrape.py` | Download Cochrane reviews as HTML/JSON |
| 2. Process | `prepare_data/process.py` | Filter by length, readability, term-preservation, and token-length |
| 3. Split | `prepare_data/split_dataset.py` | 80/10/10 train/val/test split → `{train,val,test}.{source,target,doi}` |
| 4. Token weights | `modeling/train_logr.py` | Logistic regression (abstract vs. PLS) to rank tokens by technicality |
| 5. Fine-tune | `modeling/finetune.py` | BART fine-tuning with optional unlikelihood loss |
| 6. Generate | `modeling/finetune.py --generate` | Beam-search generation of plain-language summaries |
| 7. Evaluate | `evaluate.py` | ROUGE, BLEU, BERTScore, readability, term preservation |

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for term preservation metrics

# Run the full pipeline (uses sensible defaults)
make scrape
make process
make split
make train-logr
make train
make generate
make evaluate
```

Override any path variable on the command line:

```bash
make train DATA_DIR=my/data OUTPUT_DIR=my_run
make generate OUTPUT_DIR=my_run
```

Run `make help` for the full list of targets and variables.

## Manual usage

### Fine-tuning (standard cross-entropy only)
```bash
python modeling/finetune.py \
    --model_name facebook/bart-large-cnn \
    --data_dir data/data-1024 \
    --output_dir trained_models/bart-baseline
```

### Fine-tuning with unlikelihood training
```bash
python modeling/finetune.py \
    --model_name facebook/bart-large-cnn \
    --data_dir data/data-1024 \
    --output_dir trained_models/bart-ul-both \
    --unlikelihood_training --unlikelihood_mode both \
    --cochrane_weights_file data/logr_weights/bart_freq_normalized_ids.txt \
    --newsela_weights_file  data/logr_weights/bart_freq_newsela_ids.txt \
    --unlikelihood_alpha 0.05
```

### Generation
```bash
python modeling/finetune.py --generate \
    --model_name trained_models/bart-ul-both/best_model \
    --data_dir data/data-1024 \
    --output_dir trained_models/bart-ul-both/best_model/generation \
    --num_beams 4 --length_penalty 2.0
```

### Evaluation
```bash
python evaluate.py \
    --generations_file trained_models/bart-ul-both/best_model/generation/test_generations.json \
    --output_file evaluation_results.json
```

## Repository layout

```
NLP_Proj/
├── prepare_data/
│   ├── scrape.py           # Cochrane scraper
│   ├── process.py          # filtering & token-length truncation
│   └── split_dataset.py    # 80/10/10 dataset split
├── modeling/
│   ├── finetune.py         # training + generation entry point
│   ├── train_logr.py       # logistic regression for token weights
│   └── evaluation.py       # lightweight SimplificationEvaluator class
├── evaluate.py             # comprehensive evaluation (ROUGE/BLEU/BERTScore/…)
├── Makefile                # pipeline automation
├── requirements.txt
└── README.md
```

## Key design decisions

- **Unlikelihood training**: A logistic regression classifier trained to distinguish abstract tokens from PLS tokens provides a weight vector that the cross-entropy loss is augmented with, penalising the model for generating tokens associated with technical writing.
- **`[PARA]` special token**: Paragraph boundaries in the training data are encoded as `[PARA]` tokens so BART can learn paragraph structure during generation.
- **Token-length truncation**: `process.py` truncates texts to ≤1024 BART tokens at sentence boundaries, preserving as much content as possible while staying within the model's context window.

## Tools

Python · PyTorch · HuggingFace Transformers · scikit-learn · spaCy · NLTK
