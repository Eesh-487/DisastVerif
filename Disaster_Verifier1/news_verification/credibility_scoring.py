"""
This module handles:
- NLP filtering of articles.
- Source credibility scoring.
- Duplicate article detection.
- Contextual relevance flagging.
"""

import spacy
from collections import Counter
import numpy as np

# Load spaCy model for NLP tasks (ensure en_core_web_sm is installed)
nlp = spacy.load("en_core_web_sm")

def nlp_filter_article(article):
    """
    Filter the article content using basic NLP.
    For example, ensure the content is sufficiently long.
    
    Parameters:
        article (dict): An article with a 'content' key.
    
    Returns:
        bool: True if article passes the filter, False otherwise.
    """
    content = article.get('content', '')
    # Example criteria: article must have at least 50 words.
    if len(content.split()) < 50:
        return False
    return True

def score_source_credibility(source):
    """
    Score the credibility of a news source.
    This is a placeholder. In a real application, use historical data or expert lists.
    
    Parameters:
        source (str): The news source URL or name.
    
    Returns:
        float: Credibility score between 0 and 1.
    """
    reputable_sources = ['bbc.com', 'cnn.com', 'reuters.com']
    for rep_source in reputable_sources:
        if rep_source in source:
            return 0.9
    return 0.5  # Default score for less-known sources

def detect_duplicate_articles(articles):
    """
    Remove duplicate articles based on exact title matching. 
    More advanced methods (e.g., semantic similarity) can be implemented.
    
    Parameters:
        articles (list): List of article dictionaries.
        
    Returns:
        list: List of unique articles.
    """
    seen_titles = {}
    unique_articles = []
    for article in articles:
        title = article.get('title', '').lower().strip()
        if title and title not in seen_titles:
            seen_titles[title] = article
            unique_articles.append(article)
    return unique_articles

def flag_contextual_relevance(articles, event_description):
    """
    Flag articles that are contextually relevant to a disaster event.
    Uses simple keyword matching as a placeholder.
    
    Parameters:
        articles (list): List of article dictionaries.
        event_description (str): Text description of the disaster event.
    
    Returns:
        list: Articles flagged as contextually relevant.
    """
    relevant_articles = []
    # Extract keywords from the event description (simplified method)
    event_keywords = set(event_description.lower().split())
    
    for article in articles:
        content = article.get('content', '').lower()
        content_words = set(content.split())
        common_words = event_keywords.intersection(content_words)
        # If more than 5 common words are found, consider it relevant
        if len(common_words) > 5:
            # Optionally, add a relevance score to the article.
            article['relevance_score'] = len(common_words) / len(event_keywords)
            relevant_articles.append(article)
    return relevant_articles

if __name__ == '__main__':
    # Example usage
    sample_articles = [
        {
            'title': 'Breaking News: Floods hit the city',
            'content': 'A massive flood overwhelmed the city borders. ' * 20,
            'source': 'https://www.cnn.com'
        },
        {
            'title': 'Weather update: Floods continue in the area',
            'content': 'Flooding continues as heavy rains persist. ' * 20,
            'source': 'https://www.reuters.com'
        },
        {
            'title': 'Local event: Community gathering',
            'content': 'An uplifting community event brought people together. ' * 5,
            'source': 'https://www.example-news.com'
        }
    ]
    
    # Apply NLP filtering
    filtered_articles = [article for article in sample_articles if nlp_filter_article(article)]
    print("Filtered articles count:", len(filtered_articles))
    
    # Score each article's source credibility
    for article in filtered_articles:
        article['credibility_score'] = score_source_credibility(article.get('source', ''))
    
    # Remove duplicate articles
    unique_articles = detect_duplicate_articles(filtered_articles)
    print("Unique articles count:", len(unique_articles))
    
    # Flag contextually relevant articles based on an event description
    event_description = ("severe flooding has caused significant disruption in the city "
                         "due to heavy rain and overflowing rivers")
    relevant_articles = flag_contextual_relevance(unique_articles, event_description)
    print("Contextually relevant articles count:", len(relevant_articles))
    for article in relevant_articles:
        print("Title:", article['title'], "Relevance Score:", article.get('relevance_score'))