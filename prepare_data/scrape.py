import requests
import sys
import bs4
import json
import os
import random
from os import path, makedirs
import logging
import time
from bs4 import BeautifulSoup
from math import ceil
import xml.etree.ElementTree as ET
from urllib.parse import quote, urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_doi(body):
    if body.find("a") is None:
        raise Exception("get_doi: no href found!")
    link = body.find("a")['href']
    doi = link[link.index('doi/')+4:link.index('/full')]
    return doi

def get_name(body):
    title = body.find("h3", {"class": "result-title"})
    if title is None:
        raise Exception("get_name: class result-title not found!")
    return get_text(title.find("a"))

def get_text(para):
    soup = BeautifulSoup(str(para).replace('<br/>', '\n').replace('\n ', '\n'), 'html.parser')
    text = ''.join(soup.strings).strip()
    text = text.replace('\u2010', '-')
    return text

def get_text_gen(gen):
    gen = [g.strip() for g in gen]
    text = ''.join(gen).strip()
    if len(text) > 0 and text[0] == ':':
        text = text[1:].strip()
    text = text.replace('\u2010', '-').strip()
    return text

def is_free_access(article):
    return article.find("div", {"class": "get-access-unlock"}) is None

# Check if output directory exists
output_dir = 'scraped_data/data-1024'
if not os.path.exists(output_dir):
    makedirs(output_dir)

# Load data from data_final_1024.json
try:
    with open('scraped_data/data_final_1024.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} articles from data_final_1024.json")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Shuffle data for randomized train/val/test split
random.seed(42)  # For reproducibility
random.shuffle(data)

# Calculate split sizes
num_train = int(0.8 * len(data))
num_val = int(0.1 * len(data))
split_data = {
    'train': data[:num_train],
    'val': data[num_train:num_train+num_val],
    'test': data[num_train+num_val:]
}

print(f"Split sizes: Train: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}")

def format_text_with_paragraphs(text):
    """Format text preserving paragraph structure for better training."""
    # Replace single newlines with spaces, double newlines with special token
    formatted = text.replace('\n\n', ' [PARA] ').replace('\n', ' ')
    return formatted

# Process and write data to files
for split in ['train', 'val', 'test']:
    doi_file = open(f'{output_dir}/{split}.doi', 'w', encoding='utf-8')
    source_file = open(f'{output_dir}/{split}.source', 'w', encoding='utf-8')
    target_file = open(f'{output_dir}/{split}.target', 'w', encoding='utf-8')
    
    for article in split_data[split]:
        doi = article['doi']
        
        # Use abstract_text and pls_text fields which are strings
        if 'abstract_text' in article and 'pls_text' in article:
            abstract = article['abstract_text']
            pls = article['pls_text']
        else:
            # If abstract_text/pls_text not available, extract from structure
            try:
                # Convert abstract list of sections to text
                abstract_parts = []
                for section in article['abstract']:
                    if isinstance(section, dict) and 'text' in section:
                        abstract_parts.append(section['text'])
                abstract = " ".join(abstract_parts)
                
                # Get PLS text based on type
                if article['pls_type'] == 'long':
                    pls = article['pls']
                else:
                    pls_parts = []
                    for section in article['pls']:
                        if isinstance(section, dict) and 'text' in section:
                            pls_parts.append(section['text'])
                    pls = " ".join(pls_parts)
            except Exception as e:
                print(f"Error processing article {doi}: {e}")
                continue
        
        # Write to files
        doi_file.write(doi + '\n')
        source_file.write(format_text_with_paragraphs(abstract) + '\n')
        target_file.write(format_text_with_paragraphs(pls) + '\n')
    
    doi_file.close()
    source_file.close()
    target_file.close()

print("Dataset split complete!")

def scrape_dois(results_per_page=50):
    """Scrape DOIs from Cochrane Library."""
    base_url = 'https://www.cochranelibrary.com/cdsr/reviews'
    URL = 'https://www.cochranelibrary.com/en/search?min_year=&max_year=&custom_min_year=&custom_max_year=&searchBy=6&searchText=*&selectedType=review&isWordVariations=&resultPerPage=25&searchType=basic&orderBy=relevancy&publishDateTo=&publishDateFrom=&publishYearTo=&publishYearFrom=&displayText=&forceTypeSelection=true&p_p_id=scolarissearchresultsportlet_WAR_scolarissearchresults&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-1&p_p_col_count=1&cur='
    header = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-encoding": "gzip, deflate",
        "accept-language": "en-US,en;q=0.9",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"
    }

    client = requests.Session()
    client.headers.update(header)
    client.get(base_url)

    dois = []
    try:
        soup = BeautifulSoup(client.get(base_url).text, 'html.parser')
        num_reviews = int(soup.find("span", {"class": "results-number"}).contents[0].string)
        num_search_pages = ceil(num_reviews/results_per_page)
        logger.info(f"Found {num_reviews} reviews across {num_search_pages} pages")
        
        for page in range(num_search_pages):
            logger.info(f"Scraping page {page+1} of {num_search_pages}")
            if page % 25 == 0:
                logger.info("Refreshing session to prevent timeout")
                client = requests.Session()
                client.headers.update(header)
                client.get(base_url)
                time.sleep(2)  # Add delay between requests
            
            response = client.get(base_url + str(page+1))
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Failed to get page {page+1}: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = soup.find("div", {"class": "search-results-section-body"})
            if not results:
                logger.error("No results found on page")
                continue
                
            logger.info(f"Found {len(results.contents)} items on page")
            
            for child in results.contents:
                if type(child) == bs4.element.Tag and "search-results-item" in child['class']:
                    try:
                        body = child.find("div", {"class": "search-results-item-body"})
                        if body is None:
                            raise Exception('no body!')
                        dois.append(get_doi(body))
                    except:
                        pass
    except Exception as e:
        logger.error(f"Error scraping DOIs: {e}")
        return []
        
    logger.info(f"Scraping complete. Found {len(dois)} DOIs")
    return dois

def scrape_articles(data_dir='scraped_data', results_per_page=25):
    dois = scrape_dois(results_per_page)
    scrape_articles_from_dois(dois, data_dir)

def setup_directories(data_dir, article_dir, json_dir):
    """Create directories if they don't exist."""
    for dir_path in [
        data_dir,
        path.join(data_dir, article_dir),
        path.join(data_dir, json_dir)
    ]:
        if not path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")

def get_existing_dois(data_dir, article_dir='articles'):
    """Get list of DOIs that have already been processed from HTML files."""
    existing_dois = set()
    articles_path = os.path.join(data_dir, article_dir)
    
    if os.path.exists(articles_path):
        for file in os.listdir(articles_path):
            if file.endswith('.html'):
                # Extract DOI including .pub number from filename
                doi = file[:-5]  # remove .html
                doi = doi.replace('-', '/')  # restore / in DOI
                existing_dois.add(doi)
                logger.info(f"Found existing article: {doi}")
    
    logger.info(f"Found {len(existing_dois)} existing DOIs")
    return existing_dois

def scrape_articles_from_dois(dois, data_dir):
    base_url = 'https://www.cochranelibrary.com/cdsr/reviews'
    URL = 'https://www.cochranelibrary.com/cdsr/doi/'
    header = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-encoding": "gzip, deflate",
        "accept-language": "en-US,en;q=0.9",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"
    }

    client = requests.Session()
    client.headers.update(header)
    client.get(base_url)

    article_dir = 'articles'
    json_dir = 'json'
    withdrawn_fname = 'withdrawn.txt'
    
    setup_directories(data_dir, article_dir, json_dir)

    existing_dois = get_existing_dois(data_dir)
    dois_to_scrape = [doi for doi in dois if doi not in existing_dois]
    
    logger.info(f"Total DOIs: {len(dois)}")
    logger.info(f"Already scraped: {len(existing_dois)}")
    logger.info(f"Remaining to scrape: {len(dois_to_scrape)}")
    
    for i, doi in enumerate(dois_to_scrape):
        if i > 0 and i % 50 == 0:
            client = requests.Session()
            client.headers.update(header)
            client.get(base_url)

        name = None
        try:
            soup = BeautifulSoup(client.get(URL + doi).text, 'html.parser')
            
            #write the retrieved html to a file for record-keeping purposes
            with open(path.join(data_dir, article_dir, '%s.html' % doi.replace('/', '-')), 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            #now we extract: DOI, name of article, abstract, simple summary, link
            if soup.find("h1", {"class": "publication-title"}) is None:
                raise Exception("article name not found!")

            name = soup.find("h1", {"class": "publication-title"}).string
            doc_object = {'doi': doi,
                        'name': name,
                        'free': is_free_access(soup),
                        'abstract': [], 'pls_title': None, 'pls_type': None, 'pls': []}
            
            #go heading by heading through the abstract
            abstract = soup.find("div", {"class": "full_abstract"})
            if abstract is None:
                raise Exception("abstract not found!")

            for section in abstract("section"):
                sec_object = {}
                sec_object['heading'] = get_text(section.find("h3", {"class": "title"}))
                text = [get_text(para) for para in section("p")]
                sec_object['text'] = '\n'.join(text)
                doc_object['abstract'].append(sec_object)
            
            #do the same for the plain-language summary
            pls = soup.find("div", {"class": "abstract_plainLanguageSummary"})
            if pls is None:
                raise Exception("pls not found!")
            
            doc_object['pls_title'] = get_text(pls.find("h3"))

            #determine the type of pls: "sectioned" or "long"
            if pls.find("b") is not None:
                doc_object['pls_type'] = 'sectioned'
            else:
                doc_object['pls_type'] = 'long'
            
            if doc_object['pls_type'] == 'sectioned':
    
                heading_indices = []
                texts = []

                for para in pls("p"):
                    if para.find("b") is not None:
                        heading = get_text(para.find("b"))
                        if heading[-1] == ':':
                            heading = heading[:-1]
                        texts.append(heading)
                        heading_indices.append(len(texts)-1)

                        #now grab text if there is some in the same paragraph as the heading
                        text_list = list(para.strings)
                        if len(text_list) > 1 and len(''.join(text_list[1:]).strip()) > 0:
                            text = get_text_gen(text_list[1:])
                            texts.append(text)
                    else:
                        texts.append(get_text(para))

                #edge case, if there is text before the first heading
                if heading_indices[0] > 0:
                    doc_object['pls'].append({'heading': '', 'text': '\n'.join(texts[:heading_indices[0]])})
                
                for i in range(len(heading_indices)-1):
                    doc_object['pls'].append({'heading': texts[heading_indices[i]], 'text': '\n'.join(texts[heading_indices[i]+1:heading_indices[i+1]])})

                #we know that there is at least 1 heading, so no empty list check
                doc_object['pls'].append({'heading': texts[heading_indices[-1]], 'text': '\n'.join(texts[heading_indices[-1]+1:])})

            else:
                text = [get_text(para) for para in pls("p")]
                doc_object['pls'] = '\n'.join(text)
            
            with open(path.join(data_dir, json_dir, '%s.json' % doi.replace('/', '-')), 'w', encoding='utf-8') as f:
                f.write(json.dumps(doc_object, indent=2, ensure_ascii=False))
            
            print(doi)

        except Exception as e:
            print(f'ERROR DOI {doi}: {e}')
            with open(withdrawn_fname, 'a+', encoding='utf-8') as f:
                f.write(doi + '\n')

    # now create single json file with all the articles
    articles = []
    for article_fname in os.listdir(path.join(data_dir, 'json')):
        with open(path.join(data_dir, 'json', article_fname), 'r', encoding='utf-8') as f:
            article = json.load(f)
        articles.append(article)

    with open(path.join(data_dir, 'data.json'), 'w') as f:
        f.write(json.dumps(articles, indent=2))

def scrape_pubmed_articles(data_dir='scraped_data/pubmed'):
    """Scrape PubMed articles with plain language summaries."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key = None  # Add your NCBI API key here for higher rate limits
    
    # Create directories
    setup_directories(data_dir, 'articles', 'json')
    
    # Search for articles with plain language summaries
    search_query = quote('hasplainlanguage[filter]')
    search_url = f"{base_url}/esearch.fcgi?db=pubmed&term={search_query}&retmax=1000"
    if api_key:
        search_url += f"&api_key={api_key}"
    
    logger.info("Searching PubMed for articles with plain language summaries...")
    response = requests.get(search_url)
    if response.status_code != 200:
        logger.error(f"Failed to search PubMed: {response.status_code}")
        return []
    
    # Parse PMIDs
    root = ET.fromstring(response.content)
    pmids = [id_elem.text for id_elem in root.findall(".//Id")]
    logger.info(f"Found {len(pmids)} PubMed articles")
    
    articles = []
    for i, pmid in enumerate(pmids):
        try:
            if i % 10 == 0:
                logger.info(f"Processing article {i+1}/{len(pmids)}")
            
            # Add delay to respect NCBI rate limits
            time.sleep(0.34 if not api_key else 0.1)
            
            # Fetch article
            fetch_url = f"{base_url}/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            if api_key:
                fetch_url += f"&api_key={api_key}"
            
            response = requests.get(fetch_url)
            if response.status_code != 200:
                continue
            
            # Process article and extract data
            article = process_pubmed_article(response.content, pmid)
            if article:
                articles.append(article)
                
                # Save individual JSON
                save_path = os.path.join(data_dir, 'json', f'{pmid}.json')
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(article, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing PMID {pmid}: {e}")
            continue
    
    # Save combined data
    if articles:
        combined_path = os.path.join(data_dir, 'data.json')
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2)
    
    logger.info(f"Successfully processed {len(articles)} PubMed articles")
    return articles

def process_pubmed_article(content, pmid):
    """Process PubMed article XML content."""
    article = ET.fromstring(content)
    
    # Extract abstract
    abstract_texts = []
    for abstract in article.findall(".//Abstract/AbstractText"):
        label = abstract.get('Label', '')
        text = abstract.text or ''
        if label:
            abstract_texts.append({'heading': label, 'text': text})
        else:
            abstract_texts.append({'heading': 'Abstract', 'text': text})
    
    # Extract plain language summary
    pls_elem = article.find(".//OtherAbstract[@Type='plain-language-summary']")
    if not pls_elem:
        return None
    
    pls_text = ' '.join([text.strip() for text in pls_elem.itertext() if text.strip()])
    
    return {
        'doi': f'PMID:{pmid}',
        'name': article.find(".//ArticleTitle").text,
        'abstract': abstract_texts,
        'pls_type': 'long',
        'pls': pls_text
    }

def scrape_multiple_sources():
    """Scrape articles from multiple sources."""
    # Cochrane Library
    cochrane_data = scrape_articles(data_dir='scraped_data/cochrane')
    
    # PubMed
    pubmed_data = scrape_pubmed_articles(data_dir='scraped_data/pubmed')
    
    # Combine datasets
    combined_data = cochrane_data + pubmed_data
    
    # Save combined dataset
    with open('scraped_data/combined_data.json', 'w') as f:
        json.dump(combined_data, f, indent=2)

    logger.info(f"Combined dataset created with {len(combined_data)} articles")
    return combined_data

def main():
    """Main function to run the scraper."""
    # Create output directory if it doesn't exist
    os.makedirs('scraped_data', exist_ok=True)
    
    try:
        # Scrape from multiple sources
        combined_data = scrape_multiple_sources()
        logger.info("Scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        raise

if __name__ == "__main__":
    main()