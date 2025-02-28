import json
import os
import numpy as np
from transformers import BartTokenizer
import spacy
import logging
import sys
from datetime import datetime

# Configure logging
def setup_logging():
    """Set up logging to both console and file."""
    log_filename = f'process_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    # Create logger
    logger = logging.getLogger('medical_simplification')
    logger.setLevel(logging.INFO)
    
    # Create console handler with ASCII encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def abs_length(article):
    return sum([len(x['text']) for x in article['abstract']])

def pls_length(article):
    if article['pls_type'] == 'long':
        return len(article['pls'])
    else:
        return sum([len(x['text']) for x in article['pls']])

def res_para(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    first_index = -1
    for index, sentence in enumerate(sentences):
        if any(word in sentence.lower() for word in ['journal', 'study', 'studies', 'trial']):
            first_index = index
            break
    return first_index > -1 and (index+1)/len(sentences) <= 0.5

def res_heading(heading):
    return any(word in heading.lower() for word in ['find', 'found', 'evidence', 'tell us', 'study characteristic'])

def one_para_filter(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    first_index = -1
    for index, sentence in enumerate(sentences):
        if any(word in sentence.lower() for word in ['review', 'journal', 'study', 'studies', 'paper', 'trial']):
            first_index = index
            break
    return ' '.join(sentences[first_index:]) if first_index > -1 else ''

def get_abstract_text(abstract_sections):
    """Convert abstract sections to single text string."""
    return ' '.join(section['text'] for section in abstract_sections)

def get_pls_text(article):
    """Convert PLS to single text string based on type."""
    if article['pls_type'] == 'long':
        return article['pls']
    else:  # sectioned
        return ' '.join(section['text'] for section in article['pls'])

def truncate_to_max_length(text, tokenizer, max_length=1024):
    """Simple and effective truncation to ensure we stay under token limit."""
    if not text or not isinstance(text, str):
        return ""
    
    # Direct encode-decode approach
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def create_data_json(articles_dir, output_file):
    """Create data.json from all HTML files in the articles directory."""
    logger.info(f"Creating {output_file} from HTML files in {articles_dir}")
    
    # Check if articles directory exists
    if not os.path.exists(articles_dir):
        logger.error(f"Directory {articles_dir} not found!")
        return
    
    # List all HTML files
    html_files = [f for f in os.listdir(articles_dir) if f.endswith('.html')]
    logger.info(f"Found {len(html_files)} HTML files")
    
    from bs4 import BeautifulSoup
    
    # Process each HTML file
    articles = []
    for i, html_file in enumerate(html_files):
        try:
            if i % 100 == 0:
                logger.info(f"Processing file {i+1}/{len(html_files)}")
                
            filepath = os.path.join(articles_dir, html_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Extract DOI
            doi = html_file.replace('-', '/')[:-5]  # remove .html extension
            
            # Create article object
            article = {'doi': doi, 'abstract': [], 'pls': [], 'pls_type': ''}
            
            # Extract abstract
            abstract = soup.find("div", {"class": "full_abstract"})
            if abstract:
                for section in abstract("section"):
                    sec_object = {}
                    title_elem = section.find("h3", {"class": "title"})
                    sec_object['heading'] = title_elem.text.strip() if title_elem else "Unknown"
                    text_parts = []
                    for para in section("p"):
                        text_parts.append(para.text.strip())
                    sec_object['text'] = '\n'.join(text_parts)
                    article['abstract'].append(sec_object)
            
            # Extract plain language summary
            pls = soup.find("div", {"class": "abstract_plainLanguageSummary"})
            if pls:
                pls_title = pls.find("h3")
                article['pls_title'] = pls_title.text.strip() if pls_title else ""
                
                # Determine PLS type
                if pls.find("b") is not None:
                    article['pls_type'] = 'sectioned'
                    # Extract sectioned PLS
                    sections = pls.find_all("div", {"class": "subsection"})
                    for section in sections:
                        sec_object = {}
                        heading = section.find("b")
                        sec_object['heading'] = heading.text.strip() if heading else "Unknown"
                        text_parts = []
                        for para in section("p"):
                            text_parts.append(para.text.strip())
                        sec_object['text'] = '\n'.join(text_parts)
                        article['pls'].append(sec_object)
                else:
                    article['pls_type'] = 'long'
                    # Extract long PLS
                    text_parts = []
                    for para in pls("p"):
                        text_parts.append(para.text.strip())
                    article['pls'] = '\n'.join(text_parts)
            
            # Only add if both abstract and PLS exist
            if article['abstract'] and (article['pls'] if article['pls_type'] == 'long' else article['pls']):
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error processing {html_file}: {e}")
    
    # Save to JSON
    logger.info(f"Extracted {len(articles)} valid articles")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2)
    
    return len(articles)

def clean_up_data(fname):
    logger.info(f"Starting data processing from file: {fname}")
    
    try:
        # Use explicit UTF-8 encoding when reading the file
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} articles from {fname}")
    except Exception as e:
        logger.error(f"Failed to load data from {fname}: {e}")
        return
    
    # Process abstracts
    logger.info("Processing abstracts...")
    abstract_processed = 0
    for article in data:
        first_index = -1
        for index, section in enumerate(article['abstract']):
            if 'main result' in section['heading'].strip().lower():
                first_index = index
                break
        article['abstract'] = article['abstract'][first_index:]
        abstract_processed += 1
        
    logger.info(f"Processed {abstract_processed} abstracts")
    
    # Filter by length and type
    initial_count = len(data)
    data = [x for x in data if abs_length(x) >= 500]
    logger.info(f"Filtered by length: {initial_count} → {len(data)} articles")
    
    data_long = [x for x in data if x['pls_type']=='long']
    data_sectioned = [x for x in data if x['pls_type']=='sectioned']
    logger.info(f"Split by type: {len(data_long)} long summaries, {len(data_sectioned)} sectioned summaries")

    # Split long summaries
    data_long_single = [x for x in data_long if len(x['pls'].strip().split('\n'))==1]
    data_long_multi = [x for x in data_long if len(x['pls'].strip().split('\n')) > 1]
    logger.info(f"Split long summaries: {len(data_long_single)} single paragraph, {len(data_long_multi)} multi paragraph")

    # Process single paragraph summaries
    logger.info("Processing single paragraph summaries...")
    for article in data_long_single:
        article['pls'] = one_para_filter(article['pls'])
    
    # Process multi-paragraph summaries
    logger.info("Processing multi-paragraph summaries...")
    for article in data_long_multi:
        first_index = -1
        paragraphs = article['pls'].strip().split('\n')
        for index, para in enumerate(paragraphs):
            if res_para(para):
                first_index = index
                break
        article['pls'] = '\n'.join(paragraphs[first_index:]) if first_index > -1 else ''

    # Filter empty summaries
    initial_single = len(data_long_single)
    initial_multi = len(data_long_multi)
    # data_long_single = [x for x in data_long_single if len(x['pls']) > 0]
    # data_long_multi = [x for x in data_long_multi if len(x['pls']) > 0]
    logger.info(f"Filtered empty summaries: Single {initial_single} → {len(data_long_single)}, Multi {initial_multi} → {len(data_long_multi)}")
    
    # Process sectioned summaries
    logger.info("Processing sectioned summaries...")
    for article in data_sectioned:
        first_index = -1
        for index, section in enumerate(article['pls']):
            if res_heading(section['heading']):
                first_index = index
                break
        article['pls'] = article['pls'][first_index:] if first_index > -1 else []
    
    initial_sectioned = len(data_sectioned)
    data_sectioned = [x for x in data_sectioned if len(x['pls']) > 0]
    logger.info(f"Filtered empty sectioned summaries: {initial_sectioned} → {len(data_sectioned)}")

    # Filter by length ratio
    initial_single = len(data_long_single)
    initial_multi = len(data_long_multi)
    initial_sectioned = len(data_sectioned)
    
    data_long_single = [x for x in data_long_single if (pls_length(x)/abs_length(x) >= 0.15 and pls_length(x)/abs_length(x) <= 2.0)]
    data_long_multi = [x for x in data_long_multi if (pls_length(x)/abs_length(x) >= 0.15 and pls_length(x)/abs_length(x) <= 2.0)]
    data_sectioned = [x for x in data_sectioned if (pls_length(x)/abs_length(x) >= 0.15 and pls_length(x)/abs_length(x) <= 2.0)]
    
    logger.info(f"Filtered by length ratio: Single {initial_single} → {len(data_long_single)}")
    logger.info(f"Filtered by length ratio: Multi {initial_multi} → {len(data_long_multi)}")
    logger.info(f"Filtered by length ratio: Sectioned {initial_sectioned} → {len(data_sectioned)}")
    
    # Combine all processed data
    data_final = data_long_single + data_long_multi + data_sectioned
    logger.info(f"Combined data: {len(data_final)} total articles")

    # Filter by token length
    logger.info("Loading BART tokenizer...")
    try:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return
        
    data_final_1024 = []
    
    logger.info("Processing articles for token length...")
    for i, article in enumerate(data_final):
        try:
            if i % 10 == 0:
                logger.info(f"Processing article {i+1}/{len(data_final)}")
                
            # Get text
            abstract_text = get_abstract_text(article['abstract'])
            pls_text = get_pls_text(article)
            
            # Log token lengths before truncation
            abstract_tokens = len(tokenizer.encode(abstract_text))
            pls_tokens = len(tokenizer.encode(pls_text))
            
            if abstract_tokens > 1024 or pls_tokens > 1024:
                logger.info(f"Article {i+1}: Abstract tokens: {abstract_tokens}, PLS tokens: {pls_tokens} - Truncating")
            
            # Truncate both texts
            truncated_abstract = truncate_to_max_length(abstract_text, tokenizer, max_length=1020)  # Leave buffer
            truncated_pls = truncate_to_max_length(pls_text, tokenizer, max_length=1020)  # Leave buffer
            
            # Verify truncation
            new_abstract_tokens = len(tokenizer.encode(truncated_abstract))
            new_pls_tokens = len(tokenizer.encode(truncated_pls))
            
            if new_abstract_tokens > 1024 or new_pls_tokens > 1024:
                logger.warning(f"Article {i+1}: Truncation failed - Abstract: {new_abstract_tokens}, PLS: {new_pls_tokens}")
            
            # Create new article with truncated text
            truncated_article = article.copy()
            truncated_article['abstract_text'] = truncated_abstract  # Store as plain text
            truncated_article['pls_text'] = truncated_pls  # Store as plain text
            
            # Keep original structure too
            truncated_article['original_abstract'] = article['abstract']
            truncated_article['original_pls'] = article['pls']
            
            data_final_1024.append(truncated_article)
            
        except Exception as e:
            logger.error(f"Error processing article {i+1} ({article.get('doi', 'unknown')}): {e}")
            continue

    logger.info(f"Final dataset size: {len(data_final_1024)} articles")
    
    # Save results
    logger.info("Saving results...")
    try:
        with open('scraped_data/data_final.json', 'w') as f:
            f.write(json.dumps(data_final, indent=2))
        logger.info("Saved data_final.json")
        
        with open('scraped_data/data_final_1024.json', 'w') as f:
            f.write(json.dumps(data_final_1024, indent=2))
        logger.info("Saved data_final_1024.json")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

# Add this function to process.py
def repair_data_json(fname='scraped_data/data.json'):
    """Repair corrupt data.json file by rebuilding it from individual JSON files."""
    logger.info(f"Attempting to repair {fname}...")
    
    json_dir = os.path.join(os.path.dirname(fname), 'json')
    if not os.path.exists(json_dir):
        logger.error(f"JSON directory {json_dir} not found!")
        return False
    
    try:
        # Read individual JSON files
        articles = []
        for article_fname in os.listdir(json_dir):
            if article_fname.endswith('.json'):
                with open(os.path.join(json_dir, article_fname), 'r', encoding='utf-8') as f:
                    article = json.load(f)
                articles.append(article)
        
        logger.info(f"Successfully loaded {len(articles)} articles from individual JSON files")
        
        # Write repaired data.json
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully repaired {fname}")
        return True
    except Exception as e:
        logger.error(f"Failed to repair {fname}: {e}")
        return False

# Modify the main function to use the repair function
def main():
    logger.info("Starting main processing function")
    
    # Try to repair data.json if it fails to load
    if not os.path.exists('scraped_data/data.json') or not repair_data_json('scraped_data/data.json'):
        logger.error("Could not load or repair data.json")
        return
    
    # Continue with processing
    clean_up_data('scraped_data/data.json')
    logger.info("Processing complete")

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script finished")