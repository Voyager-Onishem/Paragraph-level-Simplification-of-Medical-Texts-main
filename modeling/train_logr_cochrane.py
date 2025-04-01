import json
from joblib import dump
from os.path import join, exists
from random import shuffle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import BartTokenizer
import logging
import os
import time
import sys

# Create an absolute path for the log file
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'logr_training.log')

# Setup logging with absolute path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log the location of the log file
logger.info(f"Logging to file: {log_file}")
logger.info("Starting logistic regression training process")

def get_abstract(article):
    """Extract abstract text from an article, handling different formats."""
    logger.debug(f"Extracting abstract from article")
    
    # Add type checking and debugging
    if not isinstance(article, dict):
        logger.error(f"Article is not a dictionary: {type(article)}")
        return ""
        
    if 'abstract' not in article:
        logger.error(f"No 'abstract' field in article. Available keys: {article.keys()}")
        return ""
    
    abstract = article['abstract']
    
    # Handle different abstract formats
    if isinstance(abstract, str):
        # If abstract is already a string, return it directly
        return abstract
    elif isinstance(abstract, list):
        # If abstract is a list of dictionaries with 'text' field
        if all(isinstance(item, dict) and 'text' in item for item in abstract):
            return ' '.join([x['text'] for x in abstract])
        # If abstract is a list of strings
        elif all(isinstance(item, str) for item in abstract):
            return ' '.join(abstract)
        # Mixed or other format
        else:
            logger.warning(f"Abstract has unexpected format: {abstract[:100]}...")
            # Try to extract text anyway
            result = []
            for item in abstract:
                if isinstance(item, dict) and 'text' in item:
                    result.append(item['text'])
                elif isinstance(item, str):
                    result.append(item)
            return ' '.join(result)
    else:
        logger.error(f"Abstract has unexpected type: {type(abstract)}")
        return str(abstract)

def get_pls(article):
    """Extract Plain Language Summary text from an article, handling different formats."""
    logger.debug(f"Extracting PLS from article")
    
    # Add type checking and debugging
    if not isinstance(article, dict):
        logger.error(f"Article is not a dictionary: {type(article)}")
        return ""
        
    if 'pls' not in article:
        logger.error(f"No 'pls' field in article. Available keys: {article.keys()}")
        return ""
        
    # Get PLS type if available
    pls_type = article.get('pls_type', None)
    logger.debug(f"PLS type: {pls_type}")
    
    pls = article['pls']
    
    # Handle different PLS formats
    if isinstance(pls, str):
        # If PLS is already a string, return it directly
        return pls
    elif isinstance(pls, list):
        # If PLS is a list of dictionaries with 'text' field
        if all(isinstance(item, dict) and 'text' in item for item in pls):
            return ' '.join([x['text'] for x in pls])
        # If PLS is a list of strings
        elif all(isinstance(item, str) for item in pls):
            return ' '.join(pls)
        # Mixed or other format
        else:
            logger.warning(f"PLS has unexpected format: {pls[:100]}...")
            # Try to extract text anyway
            result = []
            for item in pls:
                if isinstance(item, dict) and 'text' in item:
                    result.append(item['text'])
                elif isinstance(item, str):
                    result.append(item)
            return ' '.join(result)
    else:
        logger.error(f"PLS has unexpected type: {type(pls)}")
        return str(pls)

def make_vector(text, tokenizer):
    logger.debug(f"Creating vector from text of length {len(text)}")
    token_ids = tokenizer.encode(text)[1:-1]  # Remove special tokens
    logger.debug(f"Encoded to {len(token_ids)} tokens")
    count_vector = np.zeros(tokenizer.vocab_size, dtype=np.int16)
    for ID in token_ids:
        count_vector[ID] += 1
    return count_vector

def construct_dataset(data, tokenizer):
    logger.info("Constructing dataset")
    start_time = time.time()
    
    if type(data) == str:
        logger.info(f"Loading data from file: {data}")
        if not exists(data):
            logger.error(f"Data file not found: {data}")
            raise FileNotFoundError(f"Could not find data file: {data}")
        try:
            data = json.load(open(data))
            
            # Debug: Inspect data structure
            logger.info(f"Data type: {type(data)}")
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    logger.info(f"First item keys: {data[0].keys()}")
                    
                    # Check the format of abstract and pls fields
                    if 'abstract' in data[0]:
                        logger.info(f"Abstract type: {type(data[0]['abstract'])}")
                        if isinstance(data[0]['abstract'], list) and len(data[0]['abstract']) > 0:
                            logger.info(f"Abstract item type: {type(data[0]['abstract'][0])}")
                    if 'pls' in data[0]:
                        logger.info(f"PLS type: {type(data[0]['pls'])}")
                        if isinstance(data[0]['pls'], list) and len(data[0]['pls']) > 0:
                            logger.info(f"PLS item type: {type(data[0]['pls'][0])}")
                    
                    # Print a sample of the first item
                    logger.info(f"Sample data: {str(data[0])[:300]}...")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from {data}")
            raise
    
    logger.info(f"Dataset contains {len(data)} articles")
    shuffle(data)
    logger.info("Data shuffled")

    X = np.empty((2*len(data), tokenizer.vocab_size), dtype=np.int16)
    y = np.empty(2*len(data), dtype=np.int16)
    
    logger.info(f"Created feature matrix of shape {X.shape}")

    index = 0
    error_count = 0
    for i, article in enumerate(data):
        try:
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(data)} articles")
            
            abstract = get_abstract(article)
            pls = get_pls(article)
            
            logger.debug(f"Article {i}: Abstract length={len(abstract)}, PLS length={len(pls)}")
            
            X[index] = make_vector(abstract, tokenizer)
            X[index+1] = make_vector(pls, tokenizer)
            y[index] = 0  # Abstract class
            y[index+1] = 1  # PLS class
            index += 2
        except Exception as e:
            logger.error(f"Error processing article {i}: {str(e)}")
            error_count += 1
            # Continue processing other articles
            continue

    if error_count > 0:
        logger.warning(f"Encountered errors in {error_count} articles")
    
    # If we skipped some articles, trim the arrays
    if index < 2*len(data):
        logger.info(f"Trimming arrays from {2*len(data)} to {index} due to errors")
        X = X[:index]
        y = y[:index]
    
    elapsed = time.time() - start_time
    logger.info(f"Dataset construction completed in {elapsed:.2f} seconds")
    logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}, Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

    return X, y

def get_vocab(tokenizer):
    logger.info("Getting vocabulary from tokenizer")
    tokens = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(tokenizer.vocab_size)]
    logger.info(f"Vocabulary size: {len(tokens)}")
    return tokens

def logr_simple_term_counts(tokenizer, save_fname, data_dir='D:/data/data_final.json', weights_dir='D:/data/logr_weights'):
    logger.info(f"Training logistic regression model with data from {data_dir}")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    start_time = time.time()
    
    X_train, y_train = construct_dataset(data_dir, tokenizer)
    logger.info(f"Dataset loaded: {X_train.shape[0]} samples with {X_train.shape[1]} features")

    logger.info("Normalizing features")
    X_train = normalize(X_train)

    logger.info("Training logistic regression model")
    model = LogisticRegression(max_iter=100)
    try:
        model.fit(X_train, y_train)
        logger.info(f"Model training completed. Score on training data: {model.score(X_train, y_train):.4f}")
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise
    
    logger.info(f"Saving model to {save_fname}")
    dump(model, save_fname)
    logger.info("Model saved successfully")

    logger.info("Extracting and sorting vocabulary weights")
    vocab = get_vocab(tokenizer)
    weights = np.squeeze(model.coef_, axis=0).tolist()
    
    logger.info("Sorting weights")
    sorted_weights = filter(lambda x: len(x[1].strip()) > 0, zip(range(tokenizer.vocab_size), vocab, weights))
    sorted_weights = list(sorted(sorted_weights, key=lambda x: x[2]))
    logger.info(f"Sorted {len(sorted_weights)} vocabulary items by weight")

    # Save weights files
    ids_path = join(weights_dir, 'bart_freq_normalized_ids.txt')
    tokens_path = join(weights_dir, 'bart_freq_normalized_tokens.txt')
    
    logger.info(f"Writing IDs weights to {ids_path}")
    with open(ids_path, 'w', encoding='utf-8') as f:
        for ID, word, weight in sorted_weights:
            f.write(f'{ID} {weight}\n')

    logger.info(f"Writing token weights to {tokens_path}")
    with open(tokens_path, 'w', encoding='utf-8') as f:
        for ID, word, weight in sorted_weights:
            f.write(f'{word} {weight}\n')
    
    # Show some analysis
    top_simple = sorted_weights[-10:]
    top_technical = sorted_weights[:10]
    
    logger.info("Top 10 words more common in simplified text:")
    for ID, word, weight in reversed(top_simple):
        logger.info(f"  {word}: {weight:.4f}")
    
    logger.info("Top 10 words more common in technical text:")
    for ID, word, weight in top_technical:
        logger.info(f"  {word}: {weight:.4f}")
    
    elapsed = time.time() - start_time
    logger.info(f"Training and weight extraction completed in {elapsed:.2f} seconds")

def list_index(l, indices):
    return [l[i] for i in indices]

def simple_kfold_term_counts(tokenizer, data_dir='D:/data/data_final.json', k=5):
    logger.info(f"Performing {k}-fold cross-validation with data from {data_dir}")
    start_time = time.time()
    
    X, y = construct_dataset(data_dir, tokenizer)
    logger.info(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    
    splitter = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies = np.zeros(k)
    
    logger.info(f"Starting {k}-fold cross-validation")
    
    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        fold_start = time.time()
        logger.info(f"Starting fold {i+1}/{k}")
        
        train_indices = train_indices.tolist()
        test_indices = test_indices.tolist()
        
        logger.info(f"Fold {i+1}: Train size={len(train_indices)}, Test size={len(test_indices)}")
        
        X_train = list_index(X, train_indices)
        y_train = list_index(y, train_indices)
        X_test = list_index(X, test_indices)
        y_test = list_index(y, test_indices)
        
        # Check class distribution
        train_dist = {0: sum(1 for y in y_train if y == 0), 1: sum(1 for y in y_train if y == 1)}
        test_dist = {0: sum(1 for y in y_test if y == 0), 1: sum(1 for y in y_test if y == 1)}
        logger.info(f"Fold {i+1} class distribution - Train: {train_dist}, Test: {test_dist}")
        
        logger.info(f"Fold {i+1}: Normalizing features")
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        
        logger.info(f"Fold {i+1}: Training model")
        model = LogisticRegression(max_iter=100)
        model.fit(X_train, y_train)
        
        logger.info(f"Fold {i+1}: Predicting on test set")
        predictions = model.predict(X_test)
        accuracies[i] = accuracy_score(y_test, predictions)
        fold_elapsed = time.time() - fold_start
        logger.info(f"Fold {i+1} accuracy: {accuracies[i]:.4f} (completed in {fold_elapsed:.2f} seconds)")
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    logger.info(f"Cross-validation results: Mean accuracy={mean_accuracy:.4f}, Std={std_accuracy:.4f}")
    logger.info(f"Individual fold accuracies: {', '.join(f'{acc:.4f}' for acc in accuracies)}")
    
    elapsed = time.time() - start_time
    logger.info(f"Cross-validation completed in {elapsed:.2f} seconds")
    
    return mean_accuracy

if __name__ == "__main__":
    try:
        logger.info("Initializing BART tokenizer")
        try:
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
            logger.info(f"Tokenizer loaded with vocabulary size: {tokenizer.vocab_size}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
        
        # Update with your actual file paths
        data_path = r"D:\Para-Level-Summ Data\data\data_final_1024.json"
        save_path = r"D:\Para-Level-Summ Data\data\logr_model\model.joblib"
        weights_path = r"D:\Para-Level-Summ Data\data\logr_weights"
        
        # Verify the data file exists
        if not exists(data_path):
            logger.error(f"Data file does not exist: {data_path}")
            # List files in the directory to help find the correct file
            try:
                data_dir = os.path.dirname(data_path)
                if exists(data_dir):
                    logger.info(f"Files in directory {data_dir}:")
                    for file in os.listdir(data_dir):
                        logger.info(f"  - {file}")
            except Exception as e:
                logger.error(f"Error listing directory: {str(e)}")
        else:
            logger.info(f"Data file found: {data_path}")
            
            try:
                logger.info("Starting model training")
                logr_simple_term_counts(tokenizer, 
                                    save_fname=save_path, 
                                    data_dir=data_path, 
                                    weights_dir=weights_path)
                logger.info("Model training completed successfully")
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
            
            try:
                logger.info("Starting cross-validation")
                simple_kfold_term_counts(tokenizer, data_dir=data_path, k=5)
                logger.info("Cross-validation completed successfully")
            except Exception as e:
                logger.error(f"Error during cross-validation: {str(e)}")
        
        logger.info("Script execution completed")
    except Exception as e:
        # Log any uncaught exceptions
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        
        # This will create a log file at the root directory as a fallback
        with open('logr_error.txt', 'a') as f:
            f.write(f"SCRIPT ERROR: {str(e)}\n")