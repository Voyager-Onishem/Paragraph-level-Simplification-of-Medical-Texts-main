import numpy as np
from sklearn.metrics import cohen_kappa_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import textstat
from bertscore import BERTScorer
import spacy

class SimplificationEvaluator:
    def __init__(self, use_spacy=True):
        self.rouge = Rouge()
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        
        # Load spaCy for entity extraction
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_md")
            except:
                self.nlp = None
                print("Warning: Could not load spaCy model. Entity metrics disabled.")
        else:
            self.nlp = None
    
    def evaluate_pair(self, source, simplified, reference=None):
        """Evaluate a single simplification against source and reference."""
        metrics = {}
        
        # 1. Content preservation metrics
        if reference:
            # ROUGE scores
            rouge_scores = self.rouge.get_scores(simplified, reference)[0]
            metrics["rouge-1"] = rouge_scores["rouge-1"]["f"]
            metrics["rouge-2"] = rouge_scores["rouge-2"]["f"] 
            metrics["rouge-l"] = rouge_scores["rouge-l"]["f"]
            
            # BLEU score
            metrics["bleu"] = sentence_bleu([reference.split()], simplified.split())
            
            # BERTScore (semantic similarity)
            P, R, F1 = self.bert_scorer.score([simplified], [reference])
            metrics["bertscore"] = F1.item()
        
        # 2. Readability metrics
        metrics["flesch_reading_ease"] = textstat.flesch_reading_ease(simplified)
        metrics["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(simplified)
        metrics["smog_index"] = textstat.smog_index(simplified)
        metrics["coleman_liau_index"] = textstat.coleman_liau_index(simplified)
        metrics["automated_readability_index"] = textstat.automated_readability_index(simplified)
        metrics["dale_chall_readability_score"] = textstat.dale_chall_readability_score(simplified)
        
        # 3. Simplification metrics
        metrics["compression_ratio"] = len(simplified.split()) / len(source.split())
        metrics["lexical_complexity_ratio"] = self._compute_lexical_complexity_ratio(source, simplified)
        
        # 4. Entity preservation (if spaCy is available)
        if self.nlp:
            metrics["entity_preservation"] = self._compute_entity_preservation(source, simplified)
        
        # 5. Syntactic simplicity metric
        metrics["avg_sentence_length"] = self._avg_sentence_length(simplified)
        
        return metrics
    
    def _compute_lexical_complexity_ratio(self, source, simplified):
        """Compute ratio of complex words."""
        source_complexity = textstat.difficult_words(source) / len(source.split())
        simplified_complexity = textstat.difficult_words(simplified) / len(simplified.split())
        return simplified_complexity / source_complexity if source_complexity > 0 else 1.0
    
    def _compute_entity_preservation(self, source, simplified):
        """Calculate how well entities from source are preserved in simplified."""
        source_doc = self.nlp(source)
        simplified_doc = self.nlp(simplified)
        
        source_entities = set([ent.text.lower() for ent in source_doc.ents])
        simplified_entities = set([ent.text.lower() for ent in simplified_doc.ents])
        
        if not source_entities:
            return 1.0
        
        return len(source_entities.intersection(simplified_entities)) / len(source_entities)
    
    def _avg_sentence_length(self, text):
        """Calculate average sentence length (words)."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0
        return np.mean([len(s.split()) for s in sentences])
    
    def evaluate_batch(self, sources, simplifieds, references=None):
        """Evaluate a batch of simplifications."""
        all_metrics = {}
        
        # Initialize metric accumulators
        for source, simplified in zip(sources[:1], simplifieds[:1]):
            sample_metrics = self.evaluate_pair(source, simplified)
            for metric in sample_metrics:
                all_metrics[metric] = []
        
        # Calculate metrics for each pair
        for i, (source, simplified) in enumerate(zip(sources, simplifieds)):
            reference = references[i] if references else None
            pair_metrics = self.evaluate_pair(source, simplified, reference)
            
            for metric, value in pair_metrics.items():
                all_metrics[metric].append(value)
        
        # Calculate mean for each metric
        return {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    def calculate_kappa(self, simplifieds, expert_simplifieds, rating_func=None):
        """Calculate Cohen's Kappa for agreement between model and expert simplifications."""
        if rating_func is None:
            # Default rating function - convert to discrete quality levels
            def rating_func(text):
                fre = textstat.flesch_reading_ease(text)
                if fre > 80:
                    return 3  # Very simple
                elif fre > 60:
                    return 2  # Moderately simple
                elif fre > 40:
                    return 1  # Somewhat simple
                else:
                    return 0  # Complex
        
        model_ratings = [rating_func(text) for text in simplifieds]
        expert_ratings = [rating_func(text) for text in expert_simplifieds]
        
        return cohen_kappa_score(model_ratings, expert_ratings)