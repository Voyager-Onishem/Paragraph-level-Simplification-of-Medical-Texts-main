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
