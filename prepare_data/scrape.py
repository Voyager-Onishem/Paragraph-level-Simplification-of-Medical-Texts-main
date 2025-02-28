import requests
import sys
import bs4
import json
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from math import ceil
import os
from os import path
import logging
import time

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
    #first replace <br> with newlines
    soup = BeautifulSoup(str(para).replace('<br/>', '\n').replace('\n ', '\n'), 'html.parser')
    text = ''.join(soup.strings).strip()

    #replace Unicode hyphen with regular '-'
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

def scrape_dois(results_per_page=50):
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

    #determine total number of reviews (first get also gives us the necessary cookies for future queries)
    soup_search_page = BeautifulSoup(client.get(base_url).text, 'html.parser')
    num_reviews = int(soup_search_page.find("span", {"class": "results-number"}).contents[0].string)
    num_reviews = 4000
    num_search_pages = ceil(num_reviews/results_per_page)
    dois = []

    logger.info(f"Starting scrape for {num_reviews} reviews across {num_search_pages} pages")
    
    #loop through the pages of results
    for page in range(num_search_pages):
        try:
            logger.info(f"Scraping page {page+1} of {num_search_pages}")
            
            if page % 25 == 0:
                logger.info("Refreshing session to prevent timeout")
                client = requests.Session()
                client.headers.update(header)
                client.get(base_url)
                time.sleep(2)  # Add delay between requests
            
            response = client.get(URL + str(page+1))
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
            logger.error(f"Error on page {page+1}: {e}")
            time.sleep(5)  # Wait longer on errors
            continue
            
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

def main():
    scrape_articles(data_dir='scraped_data', results_per_page=25)

if __name__ == "__main__":
    main()