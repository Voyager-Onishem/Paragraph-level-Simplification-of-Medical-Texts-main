#!/usr/bin/env python
# Evaluation script for medical text simplification outputs

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import spacy
import re
import logging
from typing import Dict, List, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class MedicalTextEvaluator:
    """Evaluator for medical text simplification outputs"""
    
    def __init__(self, generation_file: str, output_dir: str):
        """
        Initialize the evaluator with a file of generated summaries.
        
        Args:
            generation_file: Path to the JSON file containing generated summaries
            output_dir: Directory to save evaluation results
        """
        self.generation_file = generation_file
        self.output_dir = output_dir
        self.generations = None
        self.results = defaultdict(list)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize BERTScore if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)
        
        # Load entailment model for factual consistency
        self.load_factual_consistency_model()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_factual_consistency_model(self):
        """Load model for factual consistency evaluation"""
        try:
            model_name = "facebook/bart-large-mnli"
            self.nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.nli_model = self.nli_model.cuda()
            self.nli_model.eval()
        except Exception as e:
            logger.warning(f"Could not load factual consistency model: {e}")
            self.nli_model = None
            self.nli_tokenizer = None
    
    def load_generations(self):
        """Load the generated summaries from JSON file"""
        logger.info(f"Loading generations from {self.generation_file}")
        try:
            with open(self.generation_file, "r", encoding="utf-8") as f:
                self.generations = json.load(f)
            logger.info(f"Loaded {len(self.generations)} generations")
        except Exception as e:
            logger.error(f"Error loading generations: {e}")
            raise
    
    def evaluate_all(self):
        """Run all evaluations"""
        if self.generations is None:
            self.load_generations()
        
        logger.info("Running comprehensive evaluation...")
        
        # Iterate through all examples
        for i, example in enumerate(tqdm(self.generations, desc="Evaluating")):
            source = example["source"]
            target = example["target"]
            generated = example["generated"]
            
            # Store basic info
            self.results["idx"].append(example["idx"])
            self.results["source_length"].append(example["source_length"])
            self.results["target_length"].append(example["target_length"])
            self.results["generated_length"].append(example["generated_length"])
            
            # Calculate compression ratio
            compression_ratio = example["generated_length"] / example["source_length"]
            self.results["compression_ratio"].append(compression_ratio)
            
            # Evaluate relevance metrics
            self.evaluate_relevance(source, target, generated)
            
            # Evaluate readability metrics
            self.evaluate_readability(source, target, generated)
            
            # Evaluate lexical features
            self.evaluate_lexical_features(source, target, generated)
            
            # Evaluate factual consistency
            self.evaluate_factual_consistency(source, generated)
            
            # Print progress every 50 examples
            if (i + 1) % 50 == 0:
                logger.info(f"Evaluated {i + 1}/{len(self.generations)} examples")
        
        # Convert results to DataFrame and save
        self.save_results()
        
        # Generate summary reports and visualizations
        self.generate_reports()
        
        logger.info("Evaluation complete!")
    
    def evaluate_relevance(self, source: str, target: str, generated: str):
        """Evaluate relevance of generated text to source and reference"""
        # Calculate ROUGE scores
        rouge_scores_source = self.rouge_scorer.score(source, generated)
        rouge_scores_target = self.rouge_scorer.score(target, generated)
        
        # Store ROUGE scores against source
        self.results["rouge1_f1_source"].append(rouge_scores_source["rouge1"].fmeasure)
        self.results["rouge2_f1_source"].append(rouge_scores_source["rouge2"].fmeasure)
        self.results["rougeL_f1_source"].append(rouge_scores_source["rougeL"].fmeasure)
        
        # Store ROUGE scores against target
        self.results["rouge1_f1_target"].append(rouge_scores_target["rouge1"].fmeasure)
        self.results["rouge2_f1_target"].append(rouge_scores_target["rouge2"].fmeasure)
        self.results["rougeL_f1_target"].append(rouge_scores_target["rougeL"].fmeasure)
        
        # Calculate BERTScore
        try:
            # Calculate against source
            P_source, R_source, F1_source = self.bert_scorer.score([generated], [source])
            self.results["bertscore_f1_source"].append(F1_source.item())
            
            # Calculate against target
            P_target, R_target, F1_target = self.bert_scorer.score([generated], [target])
            self.results["bertscore_f1_target"].append(F1_target.item())
        except Exception as e:
            logger.warning(f"Error calculating BERTScore: {e}")
            self.results["bertscore_f1_source"].append(float('nan'))
            self.results["bertscore_f1_target"].append(float('nan'))
    
    def evaluate_readability(self, source: str, target: str, generated: str):
        """Evaluate readability of texts"""
        # Calculate various readability scores
        metrics = [
            "flesch_reading_ease", 
            "smog_index", 
            "flesch_kincaid_grade", 
            "coleman_liau_index",
            "automated_readability_index", 
            "dale_chall_readability_score",
            "difficult_words",
            "linsear_write_formula",
            "gunning_fog"
        ]
        
        for metric in metrics:
            # Get the method from textstat
            method = getattr(textstat, metric)
            
            # Calculate for source, target, and generated text
            source_score = method(source)
            target_score = method(target)
            generated_score = method(generated)
            
            # Store results
            self.results[f"{metric}_source"].append(source_score)
            self.results[f"{metric}_target"].append(target_score)
            self.results[f"{metric}_generated"].append(generated_score)
            
            # Calculate improvement over source (for interpretable metrics where higher is better)
            if metric == "flesch_reading_ease":
                improvement = generated_score - source_score
                self.results[f"{metric}_improvement"].append(improvement)
            # For metrics where lower is better (grade levels)
            elif "grade" in metric or "index" in metric or "score" in metric or "fog" in metric:
                improvement = source_score - generated_score
                self.results[f"{metric}_improvement"].append(improvement)
    
    def evaluate_lexical_features(self, source: str, target: str, generated: str):
        """Evaluate lexical features of texts"""
        # Tokenize texts
        source_words = word_tokenize(source.lower())
        target_words = word_tokenize(target.lower())
        generated_words = word_tokenize(generated.lower())
        
        # Calculate lexical diversity (type-token ratio)
        source_ttr = len(set(source_words)) / len(source_words) if source_words else 0
        target_ttr = len(set(target_words)) / len(target_words) if target_words else 0
        generated_ttr = len(set(generated_words)) / len(generated_words) if generated_words else 0
        
        self.results["lexical_diversity_source"].append(source_ttr)
        self.results["lexical_diversity_target"].append(target_ttr)
        self.results["lexical_diversity_generated"].append(generated_ttr)
        
        # Calculate average sentence length
        source_sents = sent_tokenize(source)
        target_sents = sent_tokenize(target)
        generated_sents = sent_tokenize(generated)
        
        source_avg_sent_len = np.mean([len(word_tokenize(s)) for s in source_sents]) if source_sents else 0
        target_avg_sent_len = np.mean([len(word_tokenize(s)) for s in target_sents]) if target_sents else 0
        generated_avg_sent_len = np.mean([len(word_tokenize(s)) for s in generated_sents]) if generated_sents else 0
        
        self.results["avg_sent_length_source"].append(source_avg_sent_len)
        self.results["avg_sent_length_target"].append(target_avg_sent_len)
        self.results["avg_sent_length_generated"].append(generated_avg_sent_len)
        
        # Calculate average word length
        source_avg_word_len = np.mean([len(w) for w in source_words]) if source_words else 0
        target_avg_word_len = np.mean([len(w) for w in target_words]) if target_words else 0
        generated_avg_word_len = np.mean([len(w) for w in generated_words]) if generated_words else 0
        
        self.results["avg_word_length_source"].append(source_avg_word_len)
        self.results["avg_word_length_target"].append(target_avg_word_len)
        self.results["avg_word_length_generated"].append(generated_avg_word_len)
        
        # Calculate sentence count
        self.results["sentence_count_source"].append(len(source_sents))
        self.results["sentence_count_target"].append(len(target_sents))
        self.results["sentence_count_generated"].append(len(generated_sents))
    
    def evaluate_factual_consistency(self, source: str, generated: str):
        """Evaluate factual consistency using NLI"""
        if self.nli_model is None or self.nli_tokenizer is None:
            self.results["factual_consistency"].append(float('nan'))
            return
        
        try:
            # Prepare source sentences
            source_sents = sent_tokenize(source)
            
            # For each source sentence, check if it's entailed by the generated text
            entailment_scores = []
            
            for premise in source_sents:
                # Skip very short sentences
                if len(premise.split()) < 3:
                    continue
                    
                with torch.no_grad():
                    inputs = self.nli_tokenizer(premise, generated, return_tensors="pt", truncation=True, max_length=512)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    outputs = self.nli_model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=1)
                    
                    # Get entailment score (index 2 for MNLI model)
                    entailment_score = predictions[:, 2].item()
                    entailment_scores.append(entailment_score)
            
            # Calculate average entailment score
            avg_score = np.mean(entailment_scores) if entailment_scores else 0
            self.results["factual_consistency"].append(avg_score)
        except Exception as e:
            logger.warning(f"Error evaluating factual consistency: {e}")
            self.results["factual_consistency"].append(float('nan'))
    
    def save_results(self):
        """Save evaluation results to CSV"""
        df = pd.DataFrame(self.results)
        output_file = os.path.join(self.output_dir, "evaluation_results.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved detailed results to {output_file}")
        
        # Also save summary statistics
        summary_df = df.describe()
        summary_file = os.path.join(self.output_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_file)
        logger.info(f"Saved summary statistics to {summary_file}")
        
        return df
    
    def generate_reports(self):
        """Generate summary reports and visualizations"""
        # Create a DataFrame from results
        df = pd.DataFrame(self.results)
        
        # Generate comparison visualizations
        self.generate_comparison_plots(df)
        
        # Generate correlation matrix
        self.generate_correlation_matrix(df)
        
        # Generate a comprehensive text report
        self.generate_text_report(df)
    
    def generate_comparison_plots(self, df):
        """Generate comparison plots between source, target, and generated texts"""
        # Set up the plotting style
        plt.style.use('ggplot')
        sns.set(font_scale=1.2)
        
        # Create readability comparison plot
        plt.figure(figsize=(12, 8))
        metrics = ['flesch_reading_ease', 'flesch_kincaid_grade', 'dale_chall_readability_score']
        labels = ['Flesch Reading Ease', 'Flesch-Kincaid Grade', 'Dale-Chall Score']
        
        # Calculate means for each metric
        means = []
        for metric in metrics:
            source_mean = df[f"{metric}_source"].mean()
            target_mean = df[f"{metric}_target"].mean()
            generated_mean = df[f"{metric}_generated"].mean()
            means.append((source_mean, target_mean, generated_mean))
        
        # Create the plot
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width, [m[0] for m in means], width, label='Source')
        ax.bar(x, [m[1] for m in means], width, label='Target')
        ax.bar(x + width, [m[2] for m in means], width, label='Generated')
        
        ax.set_ylabel('Score')
        ax.set_title('Readability Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'readability_comparison.png'))
        
        # Create ROUGE scores plot
        plt.figure(figsize=(10, 6))
        rouge_metrics = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']
        
        # Calculate means
        source_means = [df[f"{m}_source"].mean() for m in rouge_metrics]
        target_means = [df[f"{m}_target"].mean() for m in rouge_metrics]
        
        x = np.arange(len(rouge_metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, source_means, width, label='vs Source')
        ax.bar(x + width/2, target_means, width, label='vs Target')
        
        ax.set_ylabel('F1 Score')
        ax.set_title('ROUGE Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rouge_scores.png'))
        
        # Create lexical features plot
        plt.figure(figsize=(12, 8))
        lex_metrics = ['lexical_diversity', 'avg_sent_length', 'avg_word_length']
        lex_labels = ['Lexical Diversity', 'Avg. Sentence Length', 'Avg. Word Length']
        
        # Calculate means
        lex_means = []
        for metric in lex_metrics:
            source_mean = df[f"{metric}_source"].mean()
            target_mean = df[f"{metric}_target"].mean()
            generated_mean = df[f"{metric}_generated"].mean()
            lex_means.append((source_mean, target_mean, generated_mean))
        
        x = np.arange(len(lex_labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width, [m[0] for m in lex_means], width, label='Source')
        ax.bar(x, [m[1] for m in lex_means], width, label='Target')
        ax.bar(x + width, [m[2] for m in lex_means], width, label='Generated')
        
        ax.set_ylabel('Value')
        ax.set_title('Lexical Features Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(lex_labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lexical_features.png'))
    
    def generate_correlation_matrix(self, df):
        """Generate correlation matrix between metrics"""
        # Select relevant columns for correlation
        corr_columns = [
            'source_length', 'generated_length', 'compression_ratio',
            'flesch_reading_ease_generated', 'flesch_kincaid_grade_generated',
            'rouge1_f1_source', 'rouge1_f1_target', 
            'bertscore_f1_source', 'bertscore_f1_target',
            'factual_consistency'
        ]
        
        # Filter columns that exist in the DataFrame
        corr_columns = [col for col in corr_columns if col in df.columns]
        
        # Calculate correlation matrix
        corr_df = df[corr_columns].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlation Matrix of Evaluation Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'))
    
    def generate_text_report(self, df):
        """Generate a comprehensive text report"""
        report_file = os.path.join(self.output_dir, "evaluation_report.txt")
        
        with open(report_file, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("MEDICAL TEXT SIMPLIFICATION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of examples: {len(df)}\n")
            f.write(f"Average source length: {df['source_length'].mean():.2f} words\n")
            f.write(f"Average target length: {df['target_length'].mean():.2f} words\n")
            f.write(f"Average generated length: {df['generated_length'].mean():.2f} words\n")
            f.write(f"Average compression ratio: {df['compression_ratio'].mean():.2f}\n\n")
            
            # ROUGE scores
            f.write("RELEVANCE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write("ROUGE Scores (vs Source):\n")
            f.write(f"  ROUGE-1: {df['rouge1_f1_source'].mean():.4f}\n")
            f.write(f"  ROUGE-2: {df['rouge2_f1_source'].mean():.4f}\n")
            f.write(f"  ROUGE-L: {df['rougeL_f1_source'].mean():.4f}\n\n")
            
            f.write("ROUGE Scores (vs Target):\n")
            f.write(f"  ROUGE-1: {df['rouge1_f1_target'].mean():.4f}\n")
            f.write(f"  ROUGE-2: {df['rouge2_f1_target'].mean():.4f}\n")
            f.write(f"  ROUGE-L: {df['rougeL_f1_target'].mean():.4f}\n\n")
            
            # BERTScore
            if 'bertscore_f1_source' in df.columns:
                f.write("BERTScore:\n")
                f.write(f"  vs Source: {df['bertscore_f1_source'].mean():.4f}\n")
                f.write(f"  vs Target: {df['bertscore_f1_target'].mean():.4f}\n\n")
            
            # Factual consistency
            if 'factual_consistency' in df.columns:
                f.write(f"Factual Consistency: {df['factual_consistency'].mean():.4f}\n\n")
            
            # Readability metrics
            f.write("READABILITY METRICS\n")
            f.write("-" * 80 + "\n")
            
            metrics = [
                ('flesch_reading_ease', 'Flesch Reading Ease', True),
                ('flesch_kincaid_grade', 'Flesch-Kincaid Grade', False),
                ('dale_chall_readability_score', 'Dale-Chall Score', False),
                ('smog_index', 'SMOG Index', False),
                ('coleman_liau_index', 'Coleman-Liau Index', False),
                ('automated_readability_index', 'Automated Readability Index', False),
                ('gunning_fog', 'Gunning Fog Index', False)
            ]
            
            for metric, name, higher_better in metrics:
                if f"{metric}_source" in df.columns:
                    source_mean = df[f"{metric}_source"].mean()
                    target_mean = df[f"{metric}_target"].mean()
                    generated_mean = df[f"{metric}_generated"].mean()
                    
                    improvement = generated_mean - source_mean if higher_better else source_mean - generated_mean
                    
                    f.write(f"{name}:\n")
                    f.write(f"  Source: {source_mean:.2f}\n")
                    f.write(f"  Target: {target_mean:.2f}\n")
                    f.write(f"  Generated: {generated_mean:.2f}\n")
                    f.write(f"  Improvement: {improvement:.2f} ({'+' if improvement > 0 else ''}{improvement/source_mean*100:.1f}%)\n\n")
            
            # Lexical features
            f.write("LEXICAL FEATURES\n")
            f.write("-" * 80 + "\n")
            
            lex_metrics = [
                ('lexical_diversity', 'Lexical Diversity'),
                ('avg_sent_length', 'Average Sentence Length'),
                ('avg_word_length', 'Average Word Length'),
                ('sentence_count', 'Sentence Count')
            ]
            
            for metric, name in lex_metrics:
                if f"{metric}_source" in df.columns:
                    source_mean = df[f"{metric}_source"].mean()
                    target_mean = df[f"{metric}_target"].mean()
                    generated_mean = df[f"{metric}_generated"].mean()
                    
                    f.write(f"{name}:\n")
                    f.write(f"  Source: {source_mean:.2f}\n")
                    f.write(f"  Target: {target_mean:.2f}\n")
                    f.write(f"  Generated: {generated_mean:.2f}\n\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Saved comprehensive evaluation report to {report_file}")


def get_parser():
    """Get argument parser"""
    parser = argparse.ArgumentParser(description="Evaluate medical text simplification outputs")
    
    # Required arguments
    parser.add_argument(
        "--generation_file", 
        type=str, 
        required=True,
        help="Path to the JSON file containing generated summaries"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--skip_factual", 
        action="store_true",
        help="Skip factual consistency evaluation (faster)"
    )
    parser.add_argument(
        "--skip_bertscore", 
        action="store_true",
        help="Skip BERTScore calculation (faster)"
    )
    
    return parser


def main():
    """Main function"""
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MedicalTextEvaluator(
        generation_file=args.generation_file,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()