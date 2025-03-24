import logging
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import pickle
# Setting up path to ensure proper imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
from geospatial_analysis.utils import haversine_distance
from text_similarity.sbert_embeddings import generate_embeddings
from news_verification.news_scraper import get_disaster_news

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("disaster_verifier_service.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables to store loaded models
_geospatial_model = None
_text_model = None
_news_model = None
_models_loaded = False

def load_models(force_retrain=False):
    """
    Load trained models for verification
    
    Args:
        force_retrain (bool): Force retraining models even if cached versions exist
        
    Returns:
        tuple: (geospatial_model, text_model, news_model)
    """
    global _geospatial_model, _text_model, _news_model, _models_loaded
    
    # If models are already loaded and we're not forcing a retrain, return them
    if _models_loaded and not force_retrain:
        return _geospatial_model, _text_model, _news_model
    
    # Define paths for saved models
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    geo_path = os.path.join(models_dir, 'geospatial_model.pkl')
    text_path = os.path.join(models_dir, 'text_model.pkl')
    news_path = os.path.join(models_dir, 'news_model.pkl')
    
    # Check if saved models exist and load them if not forcing retrain
    if not force_retrain and os.path.exists(geo_path) and os.path.exists(text_path) and os.path.exists(news_path):
        try:
            print("Loading saved models from disk...")
            with open(geo_path, 'rb') as f:
                _geospatial_model = pickle.load(f)
            with open(text_path, 'rb') as f:
                _text_model = pickle.load(f)
            with open(news_path, 'rb') as f:
                _news_model = pickle.load(f)
            
            _models_loaded = True
            print("Models loaded successfully from disk")
            return _geospatial_model, _text_model, _news_model
        except Exception as e:
            print(f"Error loading saved models: {str(e)}. Will retrain.")
    
    # If we reach here, we need to train the models
    print("Training new models...")
    
    # Import here to avoid circular imports
    from train_model import load_data, train_geospatial_model, train_text_similarity_model, train_news_verification
    
    # Load data and train models
    try:
        df = load_data()
        if df is None or df.empty:
            print("Data loading failed or returned empty DataFrame.")
            return None, None, None
            
        # Train individual models
        _geospatial_model = train_geospatial_model(df, logger)
        _text_model = train_text_similarity_model(df, logger)
        _news_model = train_news_verification(df, logger)
        
        # Save models to disk
        try:
            with open(geo_path, 'wb') as f:
                pickle.dump(_geospatial_model, f)
            with open(text_path, 'wb') as f:
                pickle.dump(_text_model, f)
            with open(news_path, 'wb') as f:
                pickle.dump(_news_model, f)
            print("Models saved to disk for future use")
        except Exception as e:
            print(f"Error saving models: {str(e)}")
        
        _models_loaded = True
        print("Models trained successfully")
        return _geospatial_model, _text_model, _news_model
        
    except Exception as e:
        print(f"Error training models: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

def verify_disaster(disaster_data):
    """
    Verify a disaster report using trained models
    
    Args:
        disaster_data (dict): Dictionary containing disaster information
            Required keys:
            - latitude: float
            - longitude: float
            - description: str
            - location: str
            - disaster_type: str
            - date: str (format: YYYY-MM-DD)
            
    Returns:
        dict: Verification results including:
            - probability: float (0-1)
            - classification: str
            - evidence: list of supporting evidence
            - component_scores: dict of individual model scores
    """
    # Ensure required fields are present
    required_fields = ['latitude', 'longitude', 'description', 'location', 'disaster_type', 'date']
    for field in required_fields:
        if field not in disaster_data:
            return {
                'error': f"Missing required field: {field}",
                'probability': 0.0,
                'classification': "INVALID INPUT"
            }
    
    # Load models if not already loaded
    geospatial_model, text_model, news_model = load_models()
    
    # Initialize results
    geo_score = 0.0
    text_score = 0.0
    news_score = 0.0
    evidence = []
    
    # 1. GEOSPATIAL VERIFICATION
    geo_result = verify_geospatial(disaster_data, geospatial_model)
    geo_score = geo_result['score']
    if geo_result['match']:
        evidence.append(f"Found matching event within {geo_result['distance']:.2f}km")
    
    # 2. TEXT SIMILARITY VERIFICATION
    text_result = verify_text_similarity(disaster_data, text_model)
    text_score = text_result['score']
    if text_result['match'] and text_result.get('best_match_text'):
        evidence.append(f"Found similar disaster description: {text_result['best_match_text']}")
    
    # 3. NEWS VERIFICATION
    news_result = verify_news(disaster_data, news_model)
    news_score = news_result['score']
    if news_result['match']:
        evidence.append(f"Found {news_result.get('article_count', 0)} news articles about this disaster")
    
    # 4. PROBABILITY CLASSIFICATION
    # Calculate weighted confidence score
    confidence_scores = [
        {'model': 'geospatial', 'score': geo_score, 'weight': 0.4},
        {'model': 'text', 'score': text_score, 'weight': 0.3},
        {'model': 'news', 'score': news_score, 'weight': 0.3}
    ]
    
    # Calculate weighted average
    weighted_score = sum(item['score'] * item['weight'] for item in confidence_scores) / \
                      sum(item['weight'] for item in confidence_scores)
    
    # Classify disaster probability
    if weighted_score >= 0.8:
        probability_class = "HIGH PROBABILITY"
    elif weighted_score >= 0.5:
        probability_class = "MEDIUM PROBABILITY"
    elif weighted_score >= 0.3:
        probability_class = "LOW PROBABILITY"
    else:
        probability_class = "VERY LOW PROBABILITY"
    
    # Build detailed result object
    result = {
    "message": "Disaster verified and saved",
    "success": True,
    "verification": {
        "classification": probability_class
    },
    "probability": weighted_score
    }
    
    logger.info(f"Disaster verification complete: {probability_class} ({weighted_score:.4f})")
    return result

def verify_geospatial(disaster_data, geospatial_model):
    """Verify disaster using geospatial model"""
    result = {
        'match': False,
        'score': 0.0,
        'distance': float('inf'),
        'closest_event': None
    }
    
    if not geospatial_model or 'events' not in geospatial_model:
        logger.warning("Geospatial model not available for verification")
        return result
    
    input_coords = (disaster_data['latitude'], disaster_data['longitude'])
    threshold = geospatial_model.get('threshold', 100)
    closest_distance = float('inf')
    closest_event = None
    
    # Find closest event
    for event in geospatial_model['events']:
        distance = haversine_distance(input_coords, event['coords'])
        if distance < closest_distance:
            closest_distance = distance
            closest_event = event
    
    # Calculate distance-based score (inverse relationship - closer = higher score)
    geo_score = max(0, min(1, 1 - (closest_distance / (2 * threshold))))
    
    # Update result
    result['score'] = geo_score
    result['distance'] = closest_distance
    result['closest_event'] = closest_event['id'] if closest_event else None
    
    # Check if within threshold
    if closest_distance <= threshold:
        result['match'] = True
        logger.info(f"Geospatial match found: {closest_event['id']} at distance {closest_distance:.2f}km")
    else:
        logger.info(f"No geospatial match found. Closest event: {closest_event['id']} at {closest_distance:.2f}km")
    
    return result

def verify_text_similarity(disaster_data, text_model):
    """Verify disaster using text similarity model"""
    result = {
        'match': False,
        'score': 0.0,
        'best_similarity': 0,
        'best_match_text': None
    }
    
    if not text_model or 'model' not in text_model or 'embeddings' not in text_model:
        logger.warning("Text model not available for verification")
        return result
    
    # Generate embedding for input description
    input_desc = disaster_data['description']
    input_embedding = generate_embeddings(text_model['model'], [input_desc])[0]
    
    # Compare with dataset embeddings
    similarities = cosine_similarity([input_embedding], text_model['embeddings'])[0]
    
    # Find best match
    best_idx = similarities.argmax()
    best_similarity = similarities[best_idx]
    text_score = best_similarity  # Direct use of similarity as score
    
    # Update result
    result['score'] = text_score
    result['best_similarity'] = best_similarity
    
    if best_similarity >= 0.6:  # Threshold for text similarity
        result['match'] = True
        best_match_text = text_model['descriptions'][best_idx]
        result['best_match_text'] = best_match_text[:100] + "..." if len(best_match_text) > 100 else best_match_text
        logger.info(f"Text match found with similarity {best_similarity:.4f}")
    else:
        logger.info(f"No significant text match found. Best similarity: {best_similarity:.4f}")
    
    return result

def verify_news(disaster_data, news_model):
    """Verify disaster using news verification"""
    result = {
        'match': False,
        'score': 0.0,
        'article_count': 0,
        'articles': []
    }
    
    try:
        # Parse the input date
        input_date = datetime.strptime(disaster_data['date'], "%Y-%m-%d")
        
        # Calculate days between input date and today
        days_since_event = (datetime.now() - input_date).days
        
        # Set minimum search window of 7 days
        search_days = max(7, days_since_event)
        
        # Create search query from disaster type and location
        query = f"{disaster_data['disaster_type']} {disaster_data['location']}"
        logger.info(f"Searching for news articles with query: {query} within {search_days} days")

        # Get news articles (try with reduced days to get most recent)
        news_articles = get_disaster_news(query, days=3)
        
        # Update result with articles
        result['articles'] = news_articles
        result['article_count'] = len(news_articles)
        
        if news_articles:
            # Calculate news verification score based on number of articles found
            article_factor = min(1.0, len(news_articles) / 5.0)  # Scale based on number of articles (max at 5+)
            
            # Check if disaster type appears in article titles or content
            keyword_matches = 0
            for article in news_articles:
                if disaster_data['disaster_type'].lower() in article['title'].lower() or \
                   disaster_data['location'].lower() in article['title'].lower() or \
                   disaster_data['disaster_type'].lower() in article['content'].lower():
                    keyword_matches += 1
            
            keyword_factor = min(1.0, keyword_matches / max(1, len(news_articles)))
            
            # Combine scores (weighting articles and keyword matches)
            news_score = 0.4 * article_factor + 0.6 * keyword_factor
            
            # Update result
            result['score'] = news_score
            
            if news_score >= 0.5:
                result['match'] = True
                logger.info(f"News match found with {len(news_articles)} articles, score: {news_score:.4f}")
            else:
                logger.info(f"Weak news verification with score: {news_score:.4f}")
        else:
            logger.info("No news articles found for the input disaster")
    
    except Exception as e:
        logger.error(f"Error in news verification: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return result

# Example usage
if __name__ == "__main__":
    # Sample disaster input to verify
    sample_input = {
        "latitude": 19.0760, 
        "longitude": 72.8777,
        "disaster_type": "flood", 
        "description": "Severe flooding in Mumbai has affected multiple areas with water levels rising above 3 feet in low-lying regions.",
        "location": "Mumbai, India",
        "date": "2023-07-15"
    }
    
    # Verify the sample disaster
    verification_result = verify_disaster(sample_input)
    
    # Print verification results
    print("\n=== DISASTER VERIFICATION RESULTS ===")
    print(f"Input: {sample_input['disaster_type']} at ({sample_input['latitude']}, {sample_input['longitude']})")
    print(f"Location: {sample_input['location']}")
    print(f"Description: {sample_input['description']}")
    
    print(f"\nProbability: {verification_result['probability']:.4f}")
    print(f"Classification: {verification_result['classification']}")
    
    if verification_result['evidence']:
        print("\nSupporting Evidence:")
        for item in verification_result['evidence']:
            print(f"- {item}")
    
    print("\nComponent Scores:")
    for model, score in verification_result['component_scores'].items():
        print(f"- {model.capitalize()}: {score:.4f}")