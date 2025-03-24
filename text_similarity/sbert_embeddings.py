"""
This module handles text embeddings generation using Sentence-BERT (SBERT).
It provides functionality to load models and generate embeddings for disaster descriptions.
"""

import logging
from sentence_transformers import SentenceTransformer
import torch
import os

# Configure logging
logger = logging.getLogger(__name__)

# Default model to use
DEFAULT_MODEL = 'all-MiniLM-L6-v2'

def load_model(model_name=DEFAULT_MODEL, device=None):
    """
    Load a pre-trained Sentence Transformer (SBERT) model.
    
    Parameters:
        model_name (str): Name of the model to load (from Hugging Face)
        device (str): Device to use ('cpu', 'cuda', 'cuda:0', etc.) - None for auto-detection
        
    Returns:
        model: Loaded SBERT model
    """
    # Determine device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    logger.info(f"Loading SBERT model '{model_name}' on {device}")
    
    try:
        # Try to load the model
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Successfully loaded model with embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        
        # Fallback to local backup if exists
        backup_path = os.path.join("models", "backup", model_name)
        if os.path.exists(backup_path):
            logger.info(f"Attempting to load backup model from {backup_path}")
            try:
                model = SentenceTransformer(backup_path, device=device)
                logger.info("Successfully loaded backup model")
                return model
            except Exception as e2:
                logger.error(f"Error loading backup model: {str(e2)}")
                
        # If all else fails, try to load a different, more basic model
        logger.info("Attempting to load minimal fallback model")
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)
            logger.warning("Loaded minimal fallback model instead of requested model")
            return model
        except Exception as e3:
            logger.error(f"Error loading minimal fallback model: {str(e3)}")
            raise RuntimeError("Could not load any SBERT model") from e3

def generate_embeddings(model, texts):
    """
    Generate embeddings for a list of text items.
    
    Parameters:
        model: SBERT model to use
        texts (list): List of text strings to encode
        
    Returns:
        embeddings: NumPy array of embeddings
    """
    if not texts:
        logger.warning("Empty text list provided for embedding generation")
        return []
        
    try:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = model.encode(texts, show_progress_bar=False)
        logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def most_similar_texts(query_text, reference_texts, model, top_k=5):
    """
    Find the most similar texts to a query text.
    
    Parameters:
        query_text (str): Query text
        reference_texts (list): List of reference texts to compare against
        model: SBERT model to use
        top_k (int): Number of top results to return
        
    Returns:
        list: List of (index, similarity, text) tuples for the most similar texts
    """
    # Generate embedding for query text
    query_embedding = model.encode([query_text])[0]
    
    # Generate embeddings for reference texts
    reference_embeddings = model.encode(reference_texts)
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_embedding], reference_embeddings)[0]
    
    # Get top k results
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Return results
    results = []
    for idx in top_indices:
        results.append((idx, similarities[idx], reference_texts[idx]))
    
    return results

def save_model_locally(model, path):
    """
    Save the model locally for offline use.
    
    Parameters:
        model: SBERT model to save
        path (str): Path to save the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        logger.info(f"Model saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    model = load_model()
    texts = [
        "A severe flood has affected Mumbai with water levels rising rapidly in several areas.",
        "Earthquake of magnitude 6.2 hit Delhi NCR region, buildings evacuated.",
        "Cyclone warning issued for coastal areas of Tamil Nadu.",
        "Forest fires continue to rage in Uttarakhand, affecting wildlife.",
        "Heavy rainfall causes landslide in Himachal Pradesh, road connectivity affected."
    ]
    
    embeddings = generate_embeddings(model, texts)
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings[0].shape}")
    
    # Example similarity search
    query = "Flooding reported in Mumbai suburbs after heavy rain"
    results = most_similar_texts(query, texts, model, top_k=3)
    
    print(f"\nTop 3 similar texts to '{query}':")
    for idx, similarity, text in results:
        print(f"Similarity: {similarity:.4f} | Text: {text}")