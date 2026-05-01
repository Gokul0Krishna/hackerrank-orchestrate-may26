import os
import json
import time
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

CORPUS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus')
os.makedirs(CORPUS_DIR, exist_ok=True)

def scrape_hackerrank():
    """Use Playwright to fetch the JSON API to avoid 403/429 errors."""
    articles = []
    url = 'https://support.hackerrank.com/api/v2/help_center/en-us/articles.json?per_page=100'
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        while url:
            try:
                print(f"[HackerRank] Fetching API: {url}")
                page.goto(url)
                # Get the raw text from the page (which should be the JSON string)
                content = page.locator("pre").inner_text() if page.locator("pre").is_visible() else page.content()
                
                # If Zendesk wraps JSON in HTML, BeautifulSoup can extract it
                if "<html" in content.lower():
                    soup = BeautifulSoup(content, 'html.parser')
                    content = soup.get_text()

                data = json.loads(content)
                
                for article in data.get('articles', []):
                    if article.get('body'):
                        soup = BeautifulSoup(article['body'], 'html.parser')
                        articles.append({
                            'id': str(article['id']),
                            'title': article.get('title', ''),
                            'body': soup.get_text(separator=' ', strip=True),
                            'url': article.get('html_url', ''),
                            'source': 'hackerrank'
                        })
                
                url = data.get('next_page')
            except Exception as e:
                print(f" [HackerRank] Stopped at {url}: {e}")
                break
        browser.close()
    
    _save_json('hackerrank.json', articles)
def scrape_with_playwright(source_name, base_url, seed_urls, domain_filter):
    """Generic Playwright crawler for Claude and Visa."""
    articles = []
    visited = set()
    
    with sync_playwright() as p:
        # Launch browser (headless=True for speed)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        page = context.new_page()

        def crawl(url):
            if url in visited or len(visited) > 500: # Safety cap
                return
            visited.add(url)
            
            try:
                print(f"[{source_name}] Visiting: {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                # Wait a bit for JS-heavy elements
                page.wait_for_timeout(1000) 
                
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')

                # Clean noise
                for tag in soup.find_all(['nav', 'footer', 'header', 'script', 'style', 'aside']):
                    tag.decompose()

                # Extract content
                content = soup.find('article') or soup.find('main') or soup.find(id='content')
                if content:
                    title = (soup.find('h1') or soup.find('title')).get_text(strip=True)
                    text = content.get_text(separator=' ', strip=True)
                    
                    if len(text) > 150:
                        articles.append({
                            'id': url,
                            'title': title,
                            'body': text,
                            'url': url,
                            'source': source_name
                        })

                # Find links
                links = page.query_selector_all('a[href]')
                hrefs = [l.get_attribute('href') for l in links]
                
                for href in hrefs:
                    if not href: continue
                    full = urljoin(base_url, href).split('#')[0]
                    parsed = urlparse(full)
                    
                    if (parsed.netloc == domain_filter and 
                        full not in visited and 
                        not any(ext in parsed.path.lower() for ext in ['.png', '.jpg', '.pdf', '.zip'])):
                        
                        # Recursive crawl (simplified for this context)
                        if "support" in full.lower(): # Basic filter to stay in support docs
                            crawl(full)

            except Exception as e:
                print(f" [{source_name}] Error on {url}: {e}")

        for seed in seed_urls:
            crawl(seed)

        browser.close()
    
    _save_json(f'{source_name}.json', articles)

def _save_json(filename, data):
    out = os.path.join(CORPUS_DIR, filename)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[{filename}] Saved {len(data)} items.")

if __name__ == '__main__':
    print("=== Starting Playwright Scrapers ===")
    
    # 1. HackerRank (API stays the same)
    scrape_hackerrank()
    
    # 2. Claude Support
    scrape_with_playwright(
        source_name='claude',
        base_url='https://support.claude.com',
        seed_urls=['https://support.claude.com/en/'],
        domain_filter='support.claude.com'
    )
    
    # 3. Visa India Support
    scrape_with_playwright(
        source_name='visa',
        base_url='https://www.visa.co.in',
        seed_urls=[
            'https://www.visa.co.in/support.html',
            'https://www.visa.co.in/support/consumer/visa-card-benefits.html'
        ],
        domain_filter='www.visa.co.in'
    )
    
    print("=== Scraping complete ===")