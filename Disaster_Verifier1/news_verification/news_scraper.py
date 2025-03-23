"""
This module scrapes and collects relevant news articles about disasters from Indian news sources.
It uses requests and BeautifulSoup for web scraping and supports targeted searches.
"""

import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
import urllib.parse
import json
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of major Indian news sources with their base search URLs
INDIAN_NEWS_SOURCES = [
    {
        "name": "The Times of India",
        "search_url": "https://timesofindia.indiatimes.com/searchresult.cms?query={query}&searchtype=news",
        "article_selector": ".article",
        "title_selector": "h3",
        "content_selector": ".summary",
        "date_selector": ".meta-time"
    },
    {
        "name": "NDTV",
        "search_url": "https://www.ndtv.com/search?searchtext={query}",
        "article_selector": ".news_item",
        "title_selector": ".newsHdng",
        "content_selector": ".newsCont",
        "date_selector": ".posted-on"
    },
    {
        "name": "Hindustan Times",
        "search_url": "https://www.hindustantimes.com/search?q={query}",
        "article_selector": ".media",
        "title_selector": ".media-heading a",
        "content_selector": ".para-txt",
        "date_selector": ".time-dt"
    },
    {
        "name": "The Indian Express",
        "search_url": "https://indianexpress.com/?s={query}",
        "article_selector": ".article",
        "title_selector": "h2.title",
        "content_selector": ".excerpt",
        "date_selector": ".date"
    },
    {
        "name": "The Hindu",
        "search_url": "https://www.thehindu.com/search/?q={query}",
        "article_selector": ".archive-content",
        "title_selector": "a.story-card75x1-heading",
        "content_selector": ".story-card-33-sub-text",
        "date_selector": ".dateline"
    }
]

# Google News API (alternative method)
GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={query}+location:india&hl=en-IN&gl=IN&ceid=IN:en"

def scrape_news_source(source, query, days_ago=1):
    """
    Scrape a specific news source for articles related to the query.
    
    Parameters:
        source (dict): Source configuration with search URL and selectors
        query (str): Search query (e.g., "flood Bihar")
        days_ago (int): How many days back to search
        
    Returns:
        list: Articles found in this source
    """
    articles = []
    date_threshold = datetime.now() - timedelta(days=days_ago)
    
    try:
        # Format the search URL with the query
        search_url = source["search_url"].format(query=urllib.parse.quote(query))
        logger.info(f"Searching {source['name']} with URL: {search_url}")
        
        # Add random user agent to avoid blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Get the search results page
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all article elements based on the source's selectors
        for article_elem in soup.select(source["article_selector"]):
            try:
                # Extract title
                title_elem = article_elem.select_one(source["title_selector"])
                title = title_elem.get_text(strip=True) if title_elem else "No Title"
                
                # Extract content snippet
                content_elem = article_elem.select_one(source["content_selector"])
                content = content_elem.get_text(strip=True) if content_elem else "No Content"
                
                # Extract article URL
                url = title_elem.get('href') if title_elem and title_elem.get('href') else ""
                if url and not url.startswith('http'):
                    # Handle relative URLs
                    if url.startswith('/'):
                        base_url = '/'.join(search_url.split('/')[:3])
                        url = base_url + url
                    else:
                        url = search_url + url
                
                # Extract date (if available)
                date_elem = article_elem.select_one(source["date_selector"])
                date_str = date_elem.get_text(strip=True) if date_elem else ""
                
                # Check if article is within date threshold (if date is available)
                # This is a simplified example; actual date parsing would depend on format
                is_recent = True
                if date_str and "ago" not in date_str.lower():
                    # Simple approach - just check if certain keywords exist
                    # A more robust solution would parse the actual date
                    is_recent = any(word in date_str.lower() for word in ["today", "yesterday", "hour"])
                
                if is_recent:
                    articles.append({
                        'title': title,
                        'content': content,
                        'source': source['name'],
                        'url': url,
                        'date': date_str
                    })
                    logger.debug(f"Found article: {title}")
            except Exception as e:
                logger.warning(f"Error extracting article from {source['name']}: {e}")
                
        logger.info(f"Found {len(articles)} articles from {source['name']}")
    except Exception as e:
        logger.error(f"Error scraping {source['name']}: {e}")
    
    return articles

def scrape_google_news(query, days_ago=1):
    """
    Alternative method: Scrape Google News for the query
    """
    articles = []
    try:
        url = GOOGLE_NEWS_URL.format(query=urllib.parse.quote(query))
        logger.info(f"Searching Google News with URL: {url}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        
        for item in items:
            title = item.title.text if item.title else "No Title"
            description = item.description.text if item.description else "No Content"
            link = item.link.text if item.link else ""
            pub_date = item.pubDate.text if item.pubDate else ""
            
            articles.append({
                'title': title,
                'content': description,
                'source': 'Google News',
                'url': link,
                'date': pub_date
            })
            
        logger.info(f"Found {len(articles)} articles from Google News")
    except Exception as e:
        logger.error(f"Error scraping Google News: {e}")
    
    return articles

def search_disaster_news(disaster_type, location="India", days_ago=1):
    """
    Search for news about a specific disaster type in a specific location.
    
    Parameters:
        disaster_type (str): Type of disaster (flood, earthquake, etc.)
        location (str): Location to search for (default: India)
        days_ago (int): How many days back to search
        
    Returns:
        list: Combined articles from all sources
    """
    all_articles = []
    
    # Form search queries
    primary_query = f"{disaster_type} {location}"
    secondary_query = f"{disaster_type} disaster {location}"
    
    # Try each Indian news source
    for source in INDIAN_NEWS_SOURCES:
        # Add a delay to avoid being blocked
        time.sleep(random.uniform(1, 3))
        
        # Try primary query
        articles = scrape_news_source(source, primary_query, days_ago)
        all_articles.extend(articles)
        
        # If no results, try secondary query
        if not articles:
            time.sleep(random.uniform(1, 3))
            articles = scrape_news_source(source, secondary_query, days_ago)
            all_articles.extend(articles)
    
    # If we have very few articles, try Google News as a backup
    if len(all_articles) < 5:
        google_articles = scrape_google_news(primary_query, days_ago)
        # Filter out duplicates based on title similarity
        existing_titles = [a['title'].lower() for a in all_articles]
        for article in google_articles:
            if not any(article['title'].lower() in title for title in existing_titles):
                all_articles.append(article)
    
    logger.info(f"Total articles found: {len(all_articles)}")
    return all_articles

def get_disaster_news(disaster_input, location=None, days=7):
    """
    Main function to get news about a disaster.
    
    Parameters:
        disaster_input (str): Disaster type or description
        location (str): Optional location override (default: extracted from input or "India")
        days (int): Number of days to look back
        
    Returns:
        list: Articles about the disaster
    """
    # Check if disaster_input is a string or dictionary
    if isinstance(disaster_input, dict):
        # Handle dictionary input
        disaster_type = disaster_input.get('disaster_type', 'disaster')
        location = disaster_input.get('location', 'India')
    else:
        # Parse the input to extract potential disaster type and location
        disaster_types = ["flood", "earthquake", "cyclone", "hurricane", "wildfire", 
                         "landslide", "drought", "tsunami", "avalanche", "storm"]
        
        # Extract disaster type from input
        identified_disaster = next((d for d in disaster_types if d in disaster_input.lower()), "disaster")
        
        # Extract or use provided location
        if not location:
            # List of Indian states and major cities
            indian_locations = ["andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", 
                              "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", 
                              "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", 
                              "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", 
                              "tripura", "uttar pradesh", "uttarakhand", "west bengal", "delhi", "mumbai", 
                              "kolkata", "chennai", "bangalore", "hyderabad", "ahmedabad", "pune", "jaipur"]
            
            # Try to extract location from input
            location = next((loc for loc in indian_locations if loc in disaster_input.lower()), "India")
            
        disaster_type = identified_disaster
    
    logger.info(f"Searching for news about {disaster_type} in {location} for the past {days} days")
    return search_disaster_news(disaster_type, location, days)

def save_articles_to_file(articles, filename="disaster_news.json"):
    """Save the scraped articles to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(articles)} articles to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving articles to file: {e}")
        return False

# Mock function for offline testing (when web scraping is not possible)
def get_mock_disaster_news(disaster_type, location):
    """Return mock news articles for testing without internet access"""
    return [
        {
            'title': f"{disaster_type.title()} in {location}: 200 people evacuated",
            'content': f"A severe {disaster_type} has affected {location} causing widespread damage. Local authorities have evacuated 200 people from the affected areas.",
            'source': 'Mock News Source',
            'url': 'https://example.com/mock-news-1',
            'date': datetime.now().strftime("%Y-%m-%d")
        },
        {
            'title': f"Government responds to {disaster_type} crisis in {location}",
            'content': f"The government has announced emergency relief measures for the victims of the {disaster_type} in {location}. Aid workers are on the ground providing assistance.",
            'source': 'Mock News Agency',
            'url': 'https://example.com/mock-news-2',
            'date': datetime.now().strftime("%Y-%m-%d")
        }
    ]

if __name__ == '__main__':
    # Example usage
    user_input = input("Enter disaster type and location (e.g., 'flood in Bihar'): ")
    days_back = input("How many days to look back? (default: 7): ")
    days = int(days_back) if days_back.strip() else 7
    
    try:
        # Try to get real news
        articles = get_disaster_news(user_input, days=days)
        
        # If no articles found or in offline mode, use mock data
        if not articles:
            print("No articles found online. Using mock data for testing.")
            disaster_type = user_input.split()[0] if len(user_input.split()) > 0 else "disaster"
            location = user_input.split()[-1] if len(user_input.split()) > 1 else "India"
            articles = get_mock_disaster_news(disaster_type, location)
    except Exception as e:
        print(f"Error fetching news: {e}. Using mock data instead.")
        disaster_type = user_input.split()[0] if len(user_input.split()) > 0 else "disaster"
        location = user_input.split()[-1] if len(user_input.split()) > 1 else "India"
        articles = get_mock_disaster_news(disaster_type, location)
    
    # Display results
    print(f"\nFound {len(articles)} articles about the disaster")
    for i, article in enumerate(articles[:5], 1):
        print(f"\n--- Article {i} ---")
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"Date: {article.get('date', 'N/A')}")
        print(f"URL: {article.get('url', 'N/A')}")
        print(f"Content: {article['content'][:200]}...")
    
    # Save results
    if articles:
        save_articles_to_file(articles)