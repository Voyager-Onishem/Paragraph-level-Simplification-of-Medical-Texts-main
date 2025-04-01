# Define paths - Using your fresh "modelsi" directory that isn't backed up
$MODEL_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both/best_model"
$OUTPUT_DIR = "D:/Para-Level-Summ Data/trained_models/bart-cnn-ul_both/generations"
$PRETRAINED_MODEL = "facebook/bart-large-cnn"
$DATA_DIR = "D:\Para-Level-Summ Data\data\data-1024"
$CACHE_DIR = "D:\HF_Cache"

# Set cache environment variables
$env:TRANSFORMERS_CACHE = $CACHE_DIR
$env:HF_HOME = $CACHE_DIR
$env:HF_DATASETS_CACHE = $CACHE_DIR

# Create directories
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

# Set up tokenizer and model (direct Python commands avoid encoding issues)
Write-Host "Setting up model and tokenizer..." -ForegroundColor Cyan
python -c "
from transformers import BartTokenizer, BartForConditionalGeneration
import os

# Create tokenizer from pretrained
tokenizer = BartTokenizer.from_pretrained('$PRETRAINED_MODEL')
tokenizer.save_pretrained('$MODEL_DIR')
print('Tokenizer saved to $MODEL_DIR')

# Create model from pretrained 
model = BartForConditionalGeneration.from_pretrained('$PRETRAINED_MODEL')
model.save_pretrained('$MODEL_DIR')
print('Model saved to $MODEL_DIR')

# Verify files
print('\nFiles in model directory:')
files = os.listdir('$MODEL_DIR')
for f in files:
    print(f'- {f}')
"

# Create simplified evaluation module to avoid import error
$EVAL_MODULE = "c:\Users\Voyager\Paragraph-level-Simplification-of-Medical-Texts-main\evaluation.py"

Write-Host "Creating evaluation module..." -ForegroundColor Cyan
@'
"""
Simplified evaluation module for text simplification.
"""
import logging
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

class SimplificationEvaluator:
    """Simplified evaluator for text simplification tasks."""
    
    def __init__(self, use_spacy=False):
        """Initialize evaluator with minimal requirements."""
        self.use_spacy = use_spacy
    
    def evaluate_pair(self, source, simplified, reference=None):
        """Evaluate a single simplification."""
        metrics = {}
        
        # Calculate basic metrics
        source_words = len(word_tokenize(source))
        simplified_words = len(word_tokenize(simplified))
        metrics["compression_ratio"] = simplified_words / source_words if source_words > 0 else 0
        
        # Only use readability metrics that don't require additional imports
        metrics["word_count"] = simplified_words
        metrics["sentence_count"] = len(sent_tokenize(simplified))
        if metrics["sentence_count"] > 0:
            metrics["words_per_sentence"] = simplified_words / metrics["sentence_count"]
        else:
            metrics["words_per_sentence"] = 0
        
        # If reference is provided, calculate reference-based metrics
        if reference:
            reference_words = len(word_tokenize(reference))
            metrics["length_ratio"] = simplified_words / reference_words if reference_words > 0 else 0
        
        return metrics
    
    def evaluate_batch(self, sources, simplifieds, references=None):
        """Evaluate a batch of simplifications."""
        all_metrics = []
        
        # If references is not provided, use None for each example
        if references is None:
            references = [None] * len(sources)
        
        # Evaluate each example
        for i, (source, simplified, reference) in enumerate(zip(sources, simplifieds, references)):
            metrics = self.evaluate_pair(source, simplified, reference)
            all_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = sum(m[metric] for m in all_metrics) / len(all_metrics)
        
        return avg_metrics
'@ | Out-File -FilePath $EVAL_MODULE -Encoding utf8

# Run the generation
Write-Host "`nRunning generation..." -ForegroundColor Cyan

# Generate with the model
python modeling/finetune.py `
--generate `
--model_name $MODEL_DIR `
--data_dir=$DATA_DIR `
--output_dir=$OUTPUT_DIR `
--generate_mode="test" `
--num_beams=4 `
--min_length=150 `
--template_style="detailed" `
--repetition_penalty=1.2 `
--length_penalty=2.0 `
--early_stopping=False `
--temperature=0.9 `
--min_words=150

Write-Host "`nGeneration complete!" -ForegroundColor Green
Write-Host "Output saved to: $OUTPUT_DIR" -ForegroundColor Green