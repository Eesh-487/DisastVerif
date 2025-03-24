"""
This module provides text analysis functions for disaster descriptions,
including named entity recognition, sentiment analysis, and topic modeling.
"""

import logging
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configure logging
logger = logging.getLogger(__name__)

# Initialize NLP components
try:
    # Initialize spaCy for NER
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"Could not load spaCy model: {str(e)}. Will use fallback for NER.")
    nlp = None

try:
    # Initialize NLTK for sentiment analysis
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logger.warning(f"Could not initialize NLTK sentiment analyzer: {str(e)}")
    sia = None

def perform_ner(texts):
    """
    Perform Named Entity Recognition on a list of texts.
    
    Parameters:
        texts (list): List of text strings to analyze
        
    Returns:
        list: List of entity dictionaries for each text
    """
    results = []
    
    if nlp is None:
        logger.warning("NER not available, spaCy model not loaded")
        return [[] for _ in texts]
    
    try:
        logger.info(f"Performing NER on {len(texts)} texts")
        
        for text in texts:
            text_entities = []
            
            if text and isinstance(text, str):
                # Process the text with spaCy
                doc = nlp(text)
                
                # Extract entities
                for ent in doc.ents:
                    text_entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            results.append(text_entities)
            
        logger.info(f"Found entities in {sum(1 for ents in results if ents)} texts")
        return results
    except Exception as e:
        logger.error(f"Error performing NER: {str(e)}")
        return [[] for _ in texts]

def analyze_sentiment(texts):
    """
    Analyze sentiment in a list of texts.
    
    Parameters:
        texts (list): List of text strings to analyze
        
    Returns:
        list: List of sentiment dictionaries for each text
    """
    results = []
    
    if sia is None:
        logger.warning("Sentiment analysis not available, NLTK not initialized")
        return [{'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1} for _ in texts]
    
    try:
        logger.info(f"Performing sentiment analysis on {len(texts)} texts")
        
        for text in texts:
            if text and isinstance(text, str):
                # Get sentiment scores
                sentiment = sia.polarity_scores(text)
                results.append(sentiment)
            else:
                results.append({'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1})
            
        logger.info(f"Completed sentiment analysis for {len(results)} texts")
        return results
    except Exception as e:
        logger.error(f"Error performing sentiment analysis: {str(e)}")
        return [{'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1} for _ in texts]

def topic_modeling(texts, num_topics=5, num_words=10):
    """
    Identify topics in a collection of texts using LDA.
    
    Parameters:
        texts (list): List of text strings to analyze
        num_topics (int): Number of topics to identify
        num_words (int): Number of words per topic to return
        
    Returns:
        dict: Topic model results including topics and document-topic distributions
    """
    if not texts or len(texts) < 2:
        logger.warning("Not enough texts for topic modeling (min 2 required)")
        return {
            'topics': [],
            'doc_topics': []
        }
    
    try:
        logger.info(f"Performing topic modeling on {len(texts)} texts")
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.95, 
            min_df=2,
            max_features=1000,
            stop_words='english'
        )
        
        # Transform texts to TF-IDF features
        tfidf = tfidf_vectorizer.fit_transform(texts)
        
        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Check if we have enough features for the requested topics
        actual_num_topics = min(num_topics, tfidf.shape[1] - 1)
        if actual_num_topics < num_topics:
            logger.warning(f"Reducing topics from {num_topics} to {actual_num_topics} due to limited features")
            num_topics = actual_num_topics
        
        # Create and fit LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        
        # Transform TF-IDF features to topic distributions
        doc_topics = lda.fit_transform(tfidf)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            # Get top words for this topic
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            topics.append({
                'id': topic_idx,
                'words': top_words,
                'weights': topic[top_words_idx].tolist()
            })
        
        logger.info(f"Identified {len(topics)} topics")
        
        return {
            'topics': topics,
            'doc_topics': doc_topics.tolist()
        }
    except Exception as e:
        logger.error(f"Error performing topic modeling: {str(e)}")
        return {
            'topics': [],
            'doc_topics': []
        }

def extract_keywords(text, top_n=5):
    """
    Extract key phrases or keywords from text.
    
    Parameters:
        text (str): Text to analyze
        top_n (int): Number of keywords to extract
        
    Returns:
        list: List of extracted keywords
    """
    if not text or not isinstance(text, str):
        return []
    
    if nlp is None:
        logger.warning("Keyword extraction not available, spaCy model not loaded")
        return []
    
    try:
        # Process the text with spaCy
        doc = nlp(text)
        
        # Get noun chunks and named entities as potential keywords
        keywords = []
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to phrases of 3 words or less
                keywords.append(chunk.text.lower())
        
        # Add named entities
        for ent in doc.ents:
            keywords.append(ent.text.lower())
        
        # Remove duplicates and sort by length (longer first)
        keywords = sorted(set(keywords), key=len, reverse=True)
        
        return keywords[:top_n]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []

def check_text_similarity(text1, text2):
    """
    Check similarity between two texts using various methods.
    
    Parameters:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        dict: Similarity scores
    """
    if not (text1 and text2 and isinstance(text1, str) and isinstance(text2, str)):
        return {
            'token_similarity': 0.0,
            'keyword_overlap': 0.0,
            'entity_overlap': 0.0
        }
    
    try:
        # Process texts with spaCy
        if nlp is not None:
            doc1 = nlp(text1)
            doc2 = nlp(text2)
            
            # Token similarity
            token_similarity = doc1.similarity(doc2)
            
            # Extract entities
            entities1 = [ent.text.lower() for ent in doc1.ents]
            entities2 = [ent.text.lower() for ent in doc2.ents]
            
            # Calculate entity overlap
            common_entities = set(entities1).intersection(set(entities2))
            total_entities = set(entities1).union(set(entities2))
            entity_overlap = len(common_entities) / max(1, len(total_entities))
            
            # Extract keywords
            keywords1 = extract_keywords(text1)
            keywords2 = extract_keywords(text2)
            
            # Calculate keyword overlap
            common_keywords = set(keywords1).intersection(set(keywords2))
            total_keywords = set(keywords1).union(set(keywords2))
            keyword_overlap = len(common_keywords) / max(1, len(total_keywords))
            
            return {
                'token_similarity': token_similarity,
                'keyword_overlap': keyword_overlap,
                'entity_overlap': entity_overlap
            }
        else:
            # Fallback to basic word overlap if spaCy is not available
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            common_words = words1.intersection(words2)
            total_words = words1.union(words2)
            word_overlap = len(common_words) / max(1, len(total_words))
            
            return {
                'token_similarity': word_overlap,
                'keyword_overlap': word_overlap,
                'entity_overlap': 0.0
            }
    except Exception as e:
        logger.error(f"Error calculating text similarity: {str(e)}")
        return {
            'token_similarity': 0.0,
            'keyword_overlap': 0.0,
            'entity_overlap': 0.0
        }

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example texts
    disaster_texts = [
        "A severe flood has affected Mumbai with water levels rising rapidly in several areas.",
        "Earthquake of magnitude 6.2 hit Delhi NCR region, buildings evacuated.",
        "Cyclone warning issued for coastal areas of Tamil Nadu.",
        "Forest fires continue to rage in Uttarakhand, affecting wildlife.",
        "Heavy rainfall causes landslide in Himachal Pradesh, road connectivity affected."
    ]
    
    # Example NER
    entities = perform_ner(disaster_texts)
    print("\nNamed Entities:")
    for i, text_entities in enumerate(entities):
        print(f"\nText {i+1}:")
        for entity in text_entities:
            print(f"  {entity['text']} ({entity['label']})")
    
    # Example sentiment analysis
    sentiments = analyze_sentiment(disaster_texts)
    print("\nSentiment Analysis:")
    for i, sentiment in enumerate(sentiments):
        print(f"\nText {i+1}: Compound score = {sentiment['compound']:.2f}")
        print(f"  Positive: {sentiment['pos']:.2f}, Negative: {sentiment['neg']:.2f}, Neutral: {sentiment['neu']:.2f}")
    
    # Example topic modeling
    topics = topic_modeling(disaster_texts, num_topics=2)
    print("\nTopics:")
    for topic in topics['topics']:
        print(f"\nTopic {topic['id']+1}:")
        for word, weight in zip(topic['words'], topic['weights']):
            print(f"  {word} ({weight:.3f})")