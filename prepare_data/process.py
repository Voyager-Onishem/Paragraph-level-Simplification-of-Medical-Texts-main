import json
import os
import numpy as np
from transformers import BartTokenizer
import spacy
import logging
import sys
from datetime import datetime
import nltk
import argparse

# Download NLTK resources right at the start
try:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    # The error mentions punkt_tab but nltk.download('punkt_tab') doesn't work directly
    # punkt actually contains what we need
    
    # Verify punkt data is downloaded
    punkt_path = nltk.data.find('tokenizers/punkt')
    print(f"punkt installed at: {punkt_path}")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("Will use fallback sentence splitting")

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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process and filter medical articles data")
    
    # Input/output parameters
    parser.add_argument("--input_file", type=str, default="scraped_data/data.json",
                      help="Path to input data JSON file")
    parser.add_argument("--output_dir", type=str, default="scraped_data",
                      help="Directory to save output files")
    
    # Filtering parameters
    parser.add_argument("--min_abstract_length", type=int, default=500,
                      help="Minimum length of abstract text to keep")
    parser.add_argument("--min_length_ratio", type=float, default=0.15,
                      help="Minimum ratio of PLS length to abstract length")
    parser.add_argument("--max_length_ratio", type=float, default=2.0,
                      help="Maximum ratio of PLS length to abstract length")
    parser.add_argument("--min_complexity_diff", type=float, default=5.0,
                      help="Minimum reading ease difference (PLS should be easier)")
    parser.add_argument("--min_term_preservation", type=float, default=0.4,
                      help="Minimum fraction of medical terms to preserve")
    parser.add_argument("--min_paragraphs", type=int, default=2,
                      help="Minimum number of paragraphs for good structure")
    parser.add_argument("--max_paragraphs", type=int, default=7,
                      help="Maximum number of paragraphs for good structure")
    parser.add_argument("--min_compression_ratio", type=float, default=0.3,
                      help="Minimum compression ratio (target/source length)")
    parser.add_argument("--max_compression_ratio", type=float, default=0.7,
                      help="Maximum compression ratio (target/source length)")
    parser.add_argument("--max_token_length", type=int, default=1024,
                      help="Maximum token length for truncation")
    
    # Other options
    parser.add_argument("--skip_complexity", action="store_true",
                      help="Skip complexity filtering")
    parser.add_argument("--skip_term_preservation", action="store_true",
                      help="Skip term preservation filtering")
    parser.add_argument("--skip_paragraph_structure", action="store_true",
                      help="Skip paragraph structure filtering")
    parser.add_argument("--skip_length_guidance", action="store_true",
                      help="Skip length guidance filtering")
    parser.add_argument("--force_repair", action="store_true",
                      help="Force repair of data.json even if it exists")
    
    return parser.parse_args()

# Load spaCy model with fallback
try:
    logger.info("Attempting to load spaCy model en_core_web_sm...")
    nlp = spacy.load('en_core_web_sm')
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.warning(f"Failed to load spaCy model: {e}")
    logger.warning("Attempting to download a compatible model...")
    
    try:
        # Try to get a compatible version
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load('en_core_web_sm')
        logger.info("Successfully downloaded and loaded spaCy model")
    except Exception as download_err:
        logger.error(f"Could not download compatible model: {download_err}")
        logger.warning("Creating basic spaCy Language object as fallback")
        # Create a minimal Language object as fallback
        nlp = spacy.blank("en")
        
        # Define minimal sentence splitter
        def simple_sentencizer(doc):
            for i, token in enumerate(doc[:-1]):
                if token.text in ['.', '!', '?', ';'] and not doc[i+1].is_punct:
                    doc[i+1].is_sent_start = True
            return doc
        
        nlp.add_pipe("sentencizer")
        logger.warning("Using minimal sentence splitting functionality")

def abs_length(article):
    return sum([len(x['text']) for x in article['abstract']])

def pls_length(article):
    if article['pls_type'] == 'long':
        return len(article['pls'])
    else:
        return sum([len(x['text']) for x in article['pls']])

def res_para(text):
    """Identify result paragraphs with less reliance on spaCy features."""
    try:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    except:
        # Fallback to basic sentence splitting
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    first_index = -1
    for index, sentence in enumerate(sentences):
        if any(word in sentence.lower() for word in ['journal', 'study', 'studies', 'trial']):
            first_index = index
            break
    
    if first_index == -1:
        return False
    
    # Check if found early in the text
    return (first_index + 1) / max(1, len(sentences)) <= 0.5

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
        pls_parts = []
        for section in article['pls']:
            if isinstance(section, dict):
                if 'heading' in section and section['heading'] and 'text' in section:
                    pls_parts.append(f"{section['heading']}: {section['text']}")
                elif 'text' in section:
                    pls_parts.append(section['text'])
        pls = "\n\n".join(pls_parts)
        return pls

def truncate_to_max_length(text, tokenizer, max_length=1024):
    """Truncate text at sentence boundaries to fit within max_length tokens."""
    if not text:
        return ""
        
    # Check if text already fits
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) <= max_length:
        return text
        
    # Simple sentence splitting function for fallback
    def simple_split_sentences(text):
        # Split text at sentence boundaries
        sentences = []
        current = ""
        for char in text:
            current += char
            # Consider various end-of-sentence punctuation
            if char in [".", "!", "?"] and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
        # Add final sentence if it exists
        if current.strip():
            sentences.append(current.strip())
        return sentences
    
    # Try to use NLTK for sentence tokenization
    try:
        sentences = nltk.sent_tokenize(text)
        print(f"NLTK tokenization successful: {len(sentences)} sentences")
    except Exception as e:
        print(f"NLTK sentence tokenization failed: {e}")
        # Fallback to simple sentence splitting
        sentences = simple_split_sentences(text)
        print(f"Using fallback sentence splitter: {len(sentences)} sentences")
    
    # Reconstruct text sentence by sentence until we reach the limit
    truncated_text = ""
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if current_length + len(sentence_tokens) + 2 <= max_length:  # +2 for special tokens
            truncated_text += sentence + " "
            current_length += len(sentence_tokens)
        else:
            break
            
    return truncated_text.strip()

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

def clean_up_data(args):
    """Process and filter the scraped data."""
    logger.info(f"Starting data processing from file: {args.input_file}")
    
    try:
        # Use explicit UTF-8 encoding when reading the file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} articles from {args.input_file}")
    except Exception as e:
        logger.error(f"Failed to load data from {args.input_file}: {e}")
        return None
    
    # Process abstracts
    logger.info("Processing abstracts...")
    abstract_processed = 0
    for article in data:
        first_index = -1
        for index, section in enumerate(article['abstract']):
            if 'main result' in section['heading'].strip().lower():
                first_index = index
                break
        if first_index > -1:  # Only modify if 'main result' was found
            article['abstract'] = article['abstract'][first_index:]
        abstract_processed += 1
        
    logger.info(f"Processed {abstract_processed} abstracts")
    
    # Filter by length and type
    initial_count = len(data)
    data = [x for x in data if abs_length(x) >= args.min_abstract_length]
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
        filtered_text = one_para_filter(article['pls'])
        if filtered_text:  # Only update if filtering produced something
            article['pls'] = filtered_text
    
    # Process multi-paragraph summaries
    logger.info("Processing multi-paragraph summaries...")
    for article in data_long_multi:
        first_index = -1
        paragraphs = article['pls'].strip().split('\n')
        for index, para in enumerate(paragraphs):
            if res_para(para):
                first_index = index
                break
        if first_index > -1:  # Only update if a relevant paragraph was found
            article['pls'] = '\n'.join(paragraphs[first_index:])

    # Filter empty summaries
    initial_single = len(data_long_single)
    initial_multi = len(data_long_multi)
    data_long_single = [x for x in data_long_single if len(x['pls']) > 0]
    data_long_multi = [x for x in data_long_multi if len(x['pls']) > 0]
    logger.info(f"Filtered empty summaries: Single {initial_single} → {len(data_long_single)}, Multi {initial_multi} → {len(data_long_multi)}")
    
    # Process sectioned summaries
    logger.info("Processing sectioned summaries...")
    for article in data_sectioned:
        first_index = -1
        for index, section in enumerate(article['pls']):
            if res_heading(section['heading']):
                first_index = index
                break
        if first_index > -1:  # Only update if a relevant section was found
            article['pls'] = article['pls'][first_index:]
    
    initial_sectioned = len(data_sectioned)
    data_sectioned = [x for x in data_sectioned if len(x['pls']) > 0]
    logger.info(f"Filtered empty sectioned summaries: {initial_sectioned} → {len(data_sectioned)}")

    # Filter by length ratio
    initial_single = len(data_long_single)
    initial_multi = len(data_long_multi)
    initial_sectioned = len(data_sectioned)
    
    data_long_single = [x for x in data_long_single if (pls_length(x)/abs_length(x) >= args.min_length_ratio and 
                                                      pls_length(x)/abs_length(x) <= args.max_length_ratio)]
    data_long_multi = [x for x in data_long_multi if (pls_length(x)/abs_length(x) >= args.min_length_ratio and 
                                                    pls_length(x)/abs_length(x) <= args.max_length_ratio)]
    data_sectioned = [x for x in data_sectioned if (pls_length(x)/abs_length(x) >= args.min_length_ratio and 
                                                  pls_length(x)/abs_length(x) <= args.max_length_ratio)]
    
    logger.info(f"Filtered by length ratio: Single {initial_single} → {len(data_long_single)}")
    logger.info(f"Filtered by length ratio: Multi {initial_multi} → {len(data_long_multi)}")
    logger.info(f"Filtered by length ratio: Sectioned {initial_sectioned} → {len(data_sectioned)}")
    
    # Combine all processed data
    data_final = data_long_single + data_long_multi + data_sectioned
    logger.info(f"Combined data: {len(data_final)} total articles")

    # Apply advanced filtering steps
    # 1. Complexity difference
    if not args.skip_complexity:
        logger.info("Applying complexity difference filtering...")
        complexity_filtered = []
        for article in data_final:
            abstract_text = get_abstract_text(article['abstract'])
            pls_text = get_pls_text(article)
            
            # Only keep examples where PLS is at least min_complexity_diff points more readable
            complexity_diff = compute_complexity_diff(abstract_text, pls_text)
            if complexity_diff > args.min_complexity_diff:
                complexity_filtered.append(article)
        
        logger.info(f"Filtered by complexity difference: {len(data_final)} → {len(complexity_filtered)}")
        data_final = complexity_filtered
    else:
        logger.info("Skipping complexity difference filtering")
    
    # 2. Term preservation
    if not args.skip_term_preservation:
        logger.info("Applying term preservation filtering...")
        term_filtered_data = []
        for article in data_final:
            abstract_text = get_abstract_text(article['abstract'])
            pls_text = get_pls_text(article)
            
            # Only keep examples with good term preservation
            term_preservation = calculate_term_preservation(abstract_text, pls_text)
            if term_preservation >= args.min_term_preservation:
                term_filtered_data.append(article)
        
        logger.info(f"Filtered by term preservation: {len(data_final)} → {len(term_filtered_data)}")
        data_final = term_filtered_data
    else:
        logger.info("Skipping term preservation filtering")
    
    # 3. Paragraph structure
    if not args.skip_paragraph_structure:
        logger.info("Applying paragraph structure filtering...")
        structured_data = []
        for article in data_final:
            pls_text = get_pls_text(article)
            
            if has_good_paragraph_structure(pls_text, args.min_paragraphs, args.max_paragraphs):
                structured_data.append(article)
        
        logger.info(f"Filtered by paragraph structure: {len(data_final)} → {len(structured_data)}")
        data_final = structured_data
    else:
        logger.info("Skipping paragraph structure filtering")
    
    # 4. Length guidance
    if not args.skip_length_guidance:
        logger.info("Applying length guidance filtering...")
        good_length_examples = []
        for article in data_final:
            abstract_text = get_abstract_text(article['abstract'])
            pls_text = get_pls_text(article)
            
            ratio = len(pls_text.split()) / len(abstract_text.split()) if len(abstract_text.split()) > 0 else 0
            if args.min_compression_ratio <= ratio <= args.max_compression_ratio:
                good_length_examples.append(article)
        
        logger.info(f"Filtered by length guidance: {len(data_final)} → {len(good_length_examples)}")
        data_final = good_length_examples
    else:
        logger.info("Skipping length guidance filtering")

    # Filter by token length
    logger.info("Loading BART tokenizer...")
    try:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None
        
    data_final_1024 = []
    
    logger.info("Processing articles for token length...")
    for i, article in enumerate(data_final):
        try:
            if i % 10 == 0:
                logger.info(f"Processing article {i+1}/{len(data_final)}")
                
            # Get text
            abstract_text = get_abstract_text(article['abstract'])
            pls_text = get_pls_text(article)
            
            # Skip articles with empty text
            if not abstract_text or not pls_text:
                logger.warning(f"Skipping article {i+1} - empty abstract or PLS")
                continue
                
            # Log token lengths before truncation
            abstract_tokens = len(tokenizer.encode(abstract_text))
            pls_tokens = len(tokenizer.encode(pls_text))
            
            if abstract_tokens > 1024 or pls_tokens > 1024:
                logger.info(f"Article {i+1}: Abstract tokens: {abstract_tokens}, PLS tokens: {pls_tokens} - Truncating")
            
            # Truncate both texts
            truncated_abstract = truncate_to_max_length(abstract_text, tokenizer, max_length=args.max_token_length-4)  # Leave buffer
            truncated_pls = truncate_to_max_length(pls_text, tokenizer, max_length=args.max_token_length-4)  # Leave buffer
            
            # Verify truncation
            new_abstract_tokens = len(tokenizer.encode(truncated_abstract))
            new_pls_tokens = len(tokenizer.encode(truncated_pls))
            
            if new_abstract_tokens > 1024 or new_pls_tokens > 1024:
                logger.warning(f"Article {i+1}: Truncation failed - Abstract: {new_abstract_tokens}, PLS: {new_pls_tokens}")
                continue
            
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
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'data_final.json'), 'w') as f:
            f.write(json.dumps(data_final, indent=2))
        logger.info("Saved data_final.json")
        
        with open(os.path.join(output_dir, 'data_final_1024.json'), 'w') as f:
            f.write(json.dumps(data_final_1024, indent=2))
        logger.info("Saved data_final_1024.json")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return data_final_1024

def compute_complexity_diff(abstract_text, pls_text):
    """Measure the difference in complexity between source and target texts."""
    try:
        from textstat import flesch_reading_ease
        
        abstract_score = flesch_reading_ease(abstract_text)
        pls_score = flesch_reading_ease(pls_text)
        
        # Higher score means easier to read, so PLS should have a higher score
        return pls_score - abstract_score
    except Exception as e:
        logger.warning(f"Error calculating complexity difference: {e}")
        return 0  # Default to 0 if calculation fails

def calculate_term_preservation(source_text, target_text):
    """Calculate how well medical terms are preserved in the simplified text."""
    try:
        # Check if spaCy model has entity recognition
        if not nlp.has_pipe("ner"):
            # Simple fallback using word overlap for key medical terms
            source_words = set(w.lower() for w in source_text.split() if len(w) > 4)
            target_words = set(w.lower() for w in target_text.split() if len(w) > 4)
            if not source_words:
                return 1.0
            return len(target_words.intersection(source_words)) / len(source_words)
        
        source_doc = nlp(source_text)
        target_doc = nlp(target_text)
        
        # Extract entities (including medical terms)
        source_entities = set([e.text.lower() for e in source_doc.ents])
        target_entities = set([e.text.lower() for e in target_doc.ents])
        
        if not source_entities:
            return 1.0
            
        return len(target_entities.intersection(source_entities)) / len(source_entities)
    except Exception as e:
        logger.warning(f"Error calculating term preservation: {e}")
        # Simple fallback using word overlap
        source_words = set(source_text.lower().split())
        target_words = set(target_text.lower().split())
        common_words = source_words.intersection(target_words)
        
        # Avoid division by zero
        if not source_words:
            return 1.0
            
        return len(common_words) / len(source_words)

def has_good_paragraph_structure(text, min_paragraphs=2, max_paragraphs=7):
    """Check if text has a reasonable paragraph structure."""
    paragraphs = [p for p in text.split('\n') if p.strip()]
    return min_paragraphs <= len(paragraphs) <= max_paragraphs

def add_length_guidance(source_text, target_text, min_ratio=0.3, max_ratio=0.7):
    """Add explicit length guidance based on source-target ratio."""
    source_words = len(source_text.split())
    target_words = len(target_text.split())
    ratio = target_words / source_words if source_words > 0 else 0
    
    if min_ratio <= ratio <= max_ratio:  # Good length ratio
        return target_text, True
    
    return target_text, False

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

def main():
    """Main processing function with proper error handling."""
    args = parse_arguments()
    logger.info("Starting main processing function")
    logger.info(f"Arguments: {args}")
    
    # Try to repair data.json if it doesn't exist or fails to load
    if not os.path.exists(args.input_file) or args.force_repair:
        logger.warning(f"{args.input_file} not found or repair forced, attempting to repair...")
        if not repair_data_json(args.input_file):
            logger.error(f"Could not create or repair {args.input_file}")
            return
    
    # Process the data
    processed_data = clean_up_data(args)
    
    if processed_data:
        logger.info(f"Processing complete. Generated {len(processed_data)} processed articles.")
    else:
        logger.error("Processing failed.")
    
    logger.info("Processing complete")

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script finished")