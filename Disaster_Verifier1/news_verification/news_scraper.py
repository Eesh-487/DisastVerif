"""
This module scrapes and collects relevant news articles.
It uses requests and BeautifulSoup for web scraping.
"""

import requests
from bs4 import BeautifulSoup

def scrape_news(url):
    """
    Scrape news articles from the given URL.
    
    Parameters:
        url (str): URL of the news website or news page.
        
    Returns:
        articles (list): List of articles as dictionaries with keys 'title', 'content', and 'source'.
    """
    articles = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Example: extract articles within <article> tags.
        for article in soup.find_all('article'):
            title_tag = article.find('h2')
            content_tag = article.find('p')
            title = title_tag.get_text(strip=True) if title_tag else "No Title"
            content = content_tag.get_text(strip=True) if content_tag else "No Content"
            articles.append({
                'title': title,
                'content': content,
                'source': url
            })
    except Exception as e:
        print(f"Error scraping news from {url}: {e}")
    return articles

if __name__ == '__main__':
    # Example usage
    test_url = "https://www.example-news.com/latest"
    scraped_articles = scrape_news(test_url)
    for article in scraped_articles:
        print("Title:", article['title'])
        print("Content:", article['content'])
        print("-" * 40)