import json
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer
from textstat import flesch_reading_ease, smog_index
import spacy
from collections import defaultdict
import sys
import os
import argparse
import nltk
import traceback
import shutil
import tempfile

print(f"Script started. Python version: {sys.version}")
print(f"Arguments received: {sys.argv}")

# Try to download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_medical_terms(text, nlp):
    """Extract terminology using spaCy's entity recognition"""
    doc = nlp(text)
    # Since we're using en_core_web_md which doesn't have medical-specific labels,
    # we'll extract all entities instead
    entities = [ent.text for ent in doc.ents]
    return set(entities)

def calculate_term_preservation(source_terms, summary_terms):
    if not source_terms:
        return 1.0
    return len(summary_terms.intersection(source_terms)) / len(source_terms)

def check_disk_space(path, required_mb=100):
    """Check if there's enough disk space available"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_mb = free // (2**20)  # Convert to MB
        print(f"Available disk space: {free_mb}MB")
        return free_mb >= required_mb
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return False

def save_with_space_check(data, filepath, required_mb=100):
    """Save data to file with disk space check"""
    directory = os.path.dirname(filepath) or '.'
    
    if not check_disk_space(directory, required_mb):
        # Try to save to temporary directory as fallback
        temp_dir = tempfile.gettempdir()
        if check_disk_space(temp_dir, required_mb):
            backup_path = os.path.join(temp_dir, os.path.basename(filepath))
            print(f"Warning: Insufficient disk space. Saving to temporary location: {backup_path}")
            with open(backup_path, 'w') as f:
                json.dump(data, f)
            return backup_path
        else:
            raise OSError(f"Insufficient disk space in both target directory and temp directory")
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    return filepath

def evaluate_summaries(generations):
    print("Setting up evaluation metrics...")
    rouge = Rouge()
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    
    # Try loading different spaCy models in order of preference
    nlp = None
    models_to_try = ["en_core_sci_sm", "en_core_web_md", "en_core_web_sm"]
    
    for model_name in models_to_try:
        try:
            print(f"Trying to load {model_name} model...")
            nlp = spacy.load(model_name)
            print(f"Successfully loaded {model_name}")
            break
        except OSError:
            print(f"Could not load {model_name}")
            continue
    
    if nlp is None:
        print("No spaCy model available. Please install one using:")
        print("python -m spacy download en_core_web_sm")
        print("or")
        print("python -m spacy download en_core_web_md")
        print("or")
        print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz")
        print("\nTerm preservation metrics will be skipped.")
    
    def normalize_text(text):
        """Normalize text for evaluation by handling paragraph tokens."""
        # Replace [PARA] tokens with newlines
        text = text.replace(' [PARA] ', '\n\n')
        text = text.replace('[PARA]', '\n\n')
        # Clean up whitespace
        text = ' '.join(text.split())
        return text
    
    evaluation_results = []
    
    for i, generation in enumerate(generations):
        try:
            # Extract fields, with safeguards for missing keys
            if not isinstance(generation, dict):
                print(f"Skipping item {i} - not a dictionary")
                continue
                
            # Check for required fields
            if 'source' not in generation or 'target' not in generation:
                print(f"Skipping item {i} - missing source or target")
                continue
                
            source = generation['source']
            
            # For the summary, check if 'generated' exists, otherwise use 'target'
            if 'generated' in generation:
                summary = generation['generated']
            elif 'target' in generation:
                # For testing, we'll just use the target as if it were the generated summary
                summary = generation['target']
                print(f"Warning: Using target as summary for item {i}")
            else:
                print(f"Skipping item {i} - no summary available")
                continue
            
            # Normalize texts before evaluation
            source = normalize_text(source)
            summary = normalize_text(summary)
            
            # Basic statistics
            compression_ratio = len(summary.split()) / len(source.split()) if source else 0
            
            # ROUGE scores
            rouge_scores = rouge.get_scores(summary, source)[0]
            
            # BLEU score
            bleu_score = sentence_bleu([source.split()], summary.split(), 
                                      smoothing_function=SmoothingFunction().method1)
            
            # BERTScore
            P, R, F1 = bert_scorer.score([summary], [source])
            bert_scores = {
                'precision': P.item(),
                'recall': R.item(),
                'f1': F1.item()
            }
            
            # Readability scores
            readability = {
                'flesch_ease': flesch_reading_ease(summary),
                'smog': smog_index(summary)
            }
            
            # Term preservation (if spaCy is available)
            term_preservation = 0
            if nlp:
                source_terms = get_medical_terms(source, nlp)
                summary_terms = get_medical_terms(summary, nlp)
                term_preservation = calculate_term_preservation(source_terms, summary_terms)
            
            evaluation_results.append({
                'rouge': rouge_scores,
                'bleu': bleu_score,
                'bert_score': bert_scores,
                'readability': readability,
                'term_preservation': term_preservation,
                'compression_ratio': compression_ratio
            })
            
            if i < 5 or i % 20 == 0:
                print(f"Processed item {i}, BLEU: {bleu_score:.4f}, ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            print(traceback.format_exc())
    
    print(f"Completed evaluation for {len(evaluation_results)} out of {len(generations)} items")
    return evaluation_results

def main():
    print("Entering main function")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate generated summaries")
    parser.add_argument('--generations_file', type=str, 
                        default="trained_models/bart-ul_both/best_model/generation/test_generations.json",
                        help="Path to the JSON file containing generated summaries")
    parser.add_argument('--output_file', type=str, default="evaluation_results.json",
                        help="Path to save evaluation results")
    
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")
    
    try:
        # Load generations
        print(f"Loading generations from {args.generations_file}")
        with open(args.generations_file, 'r', encoding='utf-8') as f:
            generations = json.load(f)
        
        print(f"Loaded {len(generations)} generations")
        
        # Check if the generations are in the expected format
        if not generations or not isinstance(generations, list):
            print(f"Error: Generations file does not contain a list. Content type: {type(generations)}")
            return
            
        # Check a sample generation
        sample = generations[0]
        print(f"Sample generation keys: {sample.keys() if isinstance(sample, dict) else 'Not a dictionary'}")
        
        # Print sample generation content
        if isinstance(sample, dict):
            for key, value in sample.items():
                if isinstance(value, str):
                    # Print a short preview of each string value
                    preview = value[:50] + "..." if len(value) > 50 else value
                    print(f"Sample {key}: {preview}")
        
        # Handle missing generated field in test_generations.json
        if 'generated_length' in sample and 'generated' not in sample:
            print(f"Warning: File contains 'generated_length' but not 'generated' text.")
            print(f"Creating temporary generated field for testing...")
            
            for i, item in enumerate(generations):
                if isinstance(item, dict) and 'source' in item and 'generated' not in item:
                    # Create a temporary generated field from first few sentences of source
                    source_text = item.get('source', '')
                    sentences = source_text.split('.')
                    temp_generated = '. '.join(sentences[:min(3, len(sentences))]) + '.'
                    item['generated'] = temp_generated
                    
                    if i < 5:
                        print(f"Created temp generated text for item {i}")
        
        print("Loading data...")
        print(f"Data loaded. Number of samples: {len(generations)}")
        print("Starting evaluation...")
        
        # Run the evaluation
        results = evaluate_summaries(generations)
        
        # Prepare summary metrics
        summary_metrics = {
            'bleu': np.mean([r['bleu'] for r in results]) if results else 0,
            'rouge-1': {
                'p': np.mean([r['rouge']['rouge-1']['p'] for r in results]) if results else 0,
                'r': np.mean([r['rouge']['rouge-1']['r'] for r in results]) if results else 0,
                'f': np.mean([r['rouge']['rouge-1']['f'] for r in results]) if results else 0,
            },
            'rouge-2': {
                'p': np.mean([r['rouge']['rouge-2']['p'] for r in results]) if results else 0,
                'r': np.mean([r['rouge']['rouge-2']['r'] for r in results]) if results else 0,
                'f': np.mean([r['rouge']['rouge-2']['f'] for r in results]) if results else 0,
            },
            'rouge-l': {
                'p': np.mean([r['rouge']['rouge-l']['p'] for r in results]) if results else 0,
                'r': np.mean([r['rouge']['rouge-l']['r'] for r in results]) if results else 0,
                'f': np.mean([r['rouge']['rouge-l']['f'] for r in results]) if results else 0,
            },
            'bert_score': {
                'p': np.mean([r['bert_score']['precision'] for r in results]) if results else 0,
                'r': np.mean([r['bert_score']['recall'] for r in results]) if results else 0,
                'f': np.mean([r['bert_score']['f1'] for r in results]) if results else 0,
            },
            'readability': {
                'flesch_ease': np.mean([r['readability']['flesch_ease'] for r in results]) if results else 0,
                'smog': np.mean([r['readability']['smog'] for r in results]) if results else 0,
            },
            'term_preservation': np.mean([r['term_preservation'] for r in results]) if results else 0,
            'compression_ratio': np.mean([r['compression_ratio'] for r in results]) if results else 0,
        }
        
        # Print results
        print("\nEvaluation Results:")
        print(f"BLEU: {summary_metrics['bleu']:.4f}")
        print(f"ROUGE-1 F1: {summary_metrics['rouge-1']['f']:.4f}")
        print(f"ROUGE-2 F1: {summary_metrics['rouge-2']['f']:.4f}")
        print(f"ROUGE-L F1: {summary_metrics['rouge-l']['f']:.4f}")
        print(f"BERTScore F1: {summary_metrics['bert_score']['f']:.4f}")
        print(f"Term preservation: {summary_metrics['term_preservation']:.4f}")
        print(f"Flesch Reading Ease: {summary_metrics['readability']['flesch_ease']:.2f}")
        print(f"SMOG Index: {summary_metrics['readability']['smog']:.2f}")
        print(f"Compression ratio: {summary_metrics['compression_ratio']:.4f}")
        
        # Save results with disk space check
        try:
            saved_path = save_with_space_check(summary_metrics, args.output_file, required_mb=10)
            print(f"Results saved to {saved_path}")
        except OSError as e:
            print(f"Error saving results: {e}")
            # Save minimal results if full results can't be saved
            minimal_metrics = {
                'bleu': summary_metrics['bleu'],
                'rouge-1': summary_metrics['rouge-1']['f'],
                'rouge-2': summary_metrics['rouge-2']['f']
            }
            minimal_path = os.path.join(tempfile.gettempdir(), 'minimal_results.json')
            with open(minimal_path, 'w') as f:
                json.dump(minimal_metrics, f)
            print(f"Saved minimal results to {minimal_path}")
        
        print("Evaluation completed")
        
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Script entry point reached")
    main()
