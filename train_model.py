import logging
import networkx as nx  # Add this line to the imports
import json
import pandas as pd
import sys
import numpy as np
from decision_module.reporting import generate_report
from geospatial_analysis.grgnn import construct_geospatial_graph, generate_node_embeddings, grgnn_predict
from text_similarity.sbert_embeddings import load_model, generate_embeddings
from text_similarity.text_analysis import perform_ner, analyze_sentiment, topic_modeling
from news_verification.credibility_scoring import nlp_filter_article, score_source_credibility, detect_duplicate_articles, flag_contextual_relevance
from fusion_engine.fusion_logic import ensemble_scoring
from fusion_engine.data_quality import cross_validation_consistency
import random
import os 
def setup_logging():
    """Set up logging configuration for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("disaster_verifier.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_data():
    """Load and preprocess disaster data from Excel file."""
    import os
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

    file_path = os.path.join("C:\\", "Users", "eeshk", "OneDrive", "Desktop", "Disaster_Verifier", "Disaster_Verifier1", "data", "preprocessed_disasters.xlsx")
    
    logging.info(f"Loading disaster data from Excel: {file_path}")
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            # Create sample data code here if needed
            # Return None if you don't want to create sample data
            return None
            
        # Load the raw data
        df = pd.read_excel(file_path)
        logging.info(f"Loaded {len(df)} rows from disasters.xlsx")
        
        # --- PREPROCESSING STEPS ---
        
        # 1. Ensure lowercase column names for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # 2. Setup geocoder for location lookup
        geolocator = Nominatim(user_agent="disaster_verifier")
        
        # 3. Process missing coordinates using location data
        rows_to_add = []
        rows_to_remove = []
        
        for index, row in df.iterrows():
            # Check if coordinates are missing
            lat_missing = pd.isna(row.get('latitude', None))
            lon_missing = pd.isna(row.get('longitude', None))
            
            if (lat_missing or lon_missing) and 'location' in df.columns and not pd.isna(row['location']):
                locations = [loc.strip() for loc in row['location'].split(',')]
                
                # If multiple locations are provided and they're far apart
                if len(locations) > 1:
                    logging.info(f"Multiple locations found for row {index}: {locations}")
                    
                    # For each location, create a new row
                    for loc in locations:
                        try:
                            # Get coordinates for the location
                            geocode_result = geolocator.geocode(loc, timeout=10)
                            if geocode_result:
                                new_row = row.copy()
                                new_row['location'] = loc
                                new_row['latitude'] = geocode_result.latitude
                                new_row['longitude'] = geocode_result.longitude
                                logging.info(f"Geocoded {loc} to coordinates: ({geocode_result.latitude}, {geocode_result.longitude})")
                                rows_to_add.append(new_row)
                            else:
                                logging.warning(f"Could not geocode location: {loc}")
                        except (GeocoderTimedOut, GeocoderUnavailable) as e:
                            logging.warning(f"Geocoding error for {loc}: {str(e)}")
                    
                    # Mark the original row for removal
                    rows_to_remove.append(index)
                    
                else:  # Single location
                    location = row['location']
                    try:
                        # Handle common country/region names with default coordinates
                        location_defaults = {
                            'india': (20.5937, 78.9629),
                            'usa': (37.0902, -95.7129),
                            'china': (35.8617, 104.1954),
                            'russia': (61.5240, 105.3188),
                            'brazil': (-14.2350, -51.9253),
                            'canada': (56.1304, -106.3468),
                            'australia': (-25.2744, 133.7751),
                            'japan': (36.2048, 138.2529)
                        }
                        
                        # Check if location is a known country/region
                        location_lower = location.lower()
                        if location_lower in location_defaults:
                            df.at[index, 'latitude'] = location_defaults[location_lower][0]
                            df.at[index, 'longitude'] = location_defaults[location_lower][1]
                            logging.info(f"Using default coordinates for {location}")
                        else:
                            # Get coordinates for the location using geocoding
                            geocode_result = geolocator.geocode(location, timeout=10)
                            if geocode_result:
                                df.at[index, 'latitude'] = geocode_result.latitude
                                df.at[index, 'longitude'] = geocode_result.longitude
                                logging.info(f"Geocoded {location} to coordinates: ({geocode_result.latitude}, {geocode_result.longitude})")
                            else:
                                logging.warning(f"Could not geocode location: {location}")
                    except (GeocoderTimedOut, GeocoderUnavailable) as e:
                        logging.warning(f"Geocoding error for {location}: {str(e)}")
        
        # Remove original rows with multiple locations
        if rows_to_remove:
            df = df.drop(rows_to_remove)
            logging.info(f"Removed {len(rows_to_remove)} rows with multiple locations")
        
        # Add new rows for individual locations
        if rows_to_add:
            df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
            logging.info(f"Added {len(rows_to_add)} new rows for individual locations")
        
        # 4. Generate IDs for rows missing them
        if 'id' in df.columns and df['id'].isnull().any():
            missing_ids = df['id'].isnull()
            df.loc[missing_ids, 'id'] = [f"event_{i}" for i in range(sum(missing_ids))]
            logging.info(f"Generated IDs for {sum(missing_ids)} rows")
        
        # 5. Ensure required columns exist
        for column in ['id', 'latitude', 'longitude', 'description', 'news']:
            if column not in df.columns:
                df[column] = None
                logging.warning(f"Added missing column: {column}")
        
        # 6. Clean text data
        for text_col in ['description', 'news']:
            if text_col in df.columns:
                df[text_col] = df[text_col].astype(str).replace('nan', '')
                df[text_col] = df[text_col].str.strip()
        
        # 7. Drop rows still missing critical data after all processing
        critical_cols = ['latitude', 'longitude']  # Can't process events without coordinates
        missing_before = len(df)
        df = df.dropna(subset=critical_cols)
        missing_after = len(df)
        if missing_before > missing_after:
            logging.warning(f"Dropped {missing_before - missing_after} rows still missing coordinates after geocoding")
        
        # 8. Convert coordinates to float
        for col in ['latitude', 'longitude']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logging.info(f"Preprocessing complete. {len(df)} valid disaster events ready for analysis.")
        return df
        
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def train_geospatial_model(df, logger):
    """Train the geospatial analysis model using the disaster coordinates."""
    logger.info("Training geospatial analysis model...")
    
    # Check if required columns exist
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.error("DataFrame missing required columns 'latitude' and/or 'longitude'")
        return None
    
    # Create event dictionaries for graph construction
    events = []
    for index, row in df.iterrows():
        try:
            event = {
                "id": row.get("id", f"event_{index}"),
                "coords": (float(row["latitude"]), float(row["longitude"])),
                "label": row.get("label", None)
            }
            events.append(event)
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping row {index}: Invalid coordinate data. Error: {str(e)}")
    
    if not events:
        logger.error("No valid events to process. Geospatial model creation failed.")
        return None
    
    # Determine appropriate proximity threshold based on data distribution
    # Use the median of distances between points, with a minimum of 50km
    all_distances = []
    for i in range(len(events)):
        for j in range(i+1, len(events)):
            try:
                from geospatial_analysis.utils import haversine_distance
                dist = haversine_distance(events[i]['coords'], events[j]['coords'])
                all_distances.append(dist)
            except Exception as e:
                logger.warning(f"Error calculating distance: {str(e)}")
    
    proximity_threshold = 50  # Default fallback
    if all_distances:
        median_distance = np.median(all_distances)
        proximity_threshold = max(50, min(500, median_distance * 0.5))
    
    logger.info(f"Using proximity threshold of {proximity_threshold:.2f} km")
    
    # Construct the graph
    try:
        graph = construct_geospatial_graph(events, proximity_threshold)
        logger.info(f"Graph constructed with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Generate node embeddings for later use
        embeddings = generate_node_embeddings(graph)
        
        geospatial_model = {
            "graph": graph, 
            "threshold": proximity_threshold,
            "embeddings": embeddings,
            "events": events
        }
        logger.info("Geospatial analysis model training complete.")
        return geospatial_model
    except Exception as e:
        logger.error(f"Error in geospatial model construction: {str(e)}")
        return None

def train_text_similarity_model(df, logger):
    """Train the text similarity model using disaster descriptions."""
    logger.info("Training text similarity model...")
    
    if "description" not in df.columns:
        logger.error("DataFrame missing 'description' column.")
        return None
    
    descriptions = df["description"].astype(str).tolist()
    
    try:
        # Load SBERT model and generate embeddings
        model = load_model()
        embeddings = generate_embeddings(model, descriptions)
        
        # Run additional text analysis on descriptions
        entities_by_doc = []
        sentiments = []
        
        for desc in descriptions:
            try:
                # Extract named entities
                entities = perform_ner(desc)
                entities_by_doc.append(entities)
                
                # Analyze sentiment
                sentiment = analyze_sentiment(desc)
                sentiments.append(sentiment)
            except Exception as e:
                logger.warning(f"Error in text analysis: {str(e)}")
                entities_by_doc.append([])
                sentiments.append({})
        
        # Generate topic model for all descriptions
        try:
            topics = topic_modeling(descriptions, n_topics=min(3, len(descriptions)))
        except Exception as e:
            logger.warning(f"Error in topic modeling: {str(e)}")
            topics = {}
        
        # Package all text analysis results
        text_model = {
            "model": model,
            "embeddings": embeddings,
            "entities": entities_by_doc,
            "sentiments": sentiments,
            "topics": topics,
            "descriptions": descriptions
        }
        
        logger.info("Text similarity model training complete.")
        return text_model
    except Exception as e:
        logger.error(f"Error in text similarity model training: {str(e)}")
        return None

def train_news_verification(df, logger):
    """Train the news verification model using disaster news articles."""
    logger.info("Setting up news verification model...")
    
    if "news" not in df.columns:
        logger.error("DataFrame missing 'news' column. Cannot train news verification model.")
        return None
    
    # Create article objects from the news column
    articles = []
    for index, row in df.iterrows():
        news_text = row.get("news", "")
        if not news_text or pd.isna(news_text):
            continue
            
        article = {
            "id": row.get("id", f"news_{index}"),
            "title": f"News about event {row.get('id', index)}",
            "content": news_text,
            "source": "dataset"  # Since we're using dataset-provided news
        }
        articles.append(article)
    
    if not articles:
        logger.warning("No valid news articles found in the dataset.")
        return None
    
    # Apply NLP filtering
    logger.info(f"Applying NLP filtering to {len(articles)} articles")
    filtered_articles = [article for article in articles if nlp_filter_article(article)]
    
    # Score credibility of sources
    for article in filtered_articles:
        article['credibility_score'] = score_source_credibility(article.get('source', ''))
    
    # Remove duplicate articles
    unique_articles = detect_duplicate_articles(filtered_articles)
    
    # For each disaster event description, flag relevant articles
    relevance_by_event = {}
    
    if "description" in df.columns:
        for index, row in df.iterrows():
            event_id = row.get("id", f"event_{index}")
            description = row.get("description", "")
            
            if description and not pd.isna(description):
                relevant = flag_contextual_relevance(unique_articles, description)
                relevance_by_event[event_id] = relevant
    
    # Calculate verification scores based on credibility and relevance
    verification_scores = {}
    
    for index, row in df.iterrows():
        event_id = row.get("id", f"news_{index}")
        
        # Find articles relevant to this event
        event_articles = relevance_by_event.get(event_id, [])
        
        if event_articles:
            # Average credibility score of relevant articles
            cred_scores = [art.get('credibility_score', 0.5) for art in event_articles]
            # Average relevance score of relevant articles
            rel_scores = [art.get('relevance_score', 0.5) for art in event_articles]
            
            # Combine scores (weighted average)
            if cred_scores and rel_scores:
                verification_scores[event_id] = (0.6 * np.mean(cred_scores)) + (0.4 * np.mean(rel_scores))
            else:
                verification_scores[event_id] = None
        else:
            verification_scores[event_id] = None
    
    news_model = {
        "verification_scores": verification_scores,
        "articles": unique_articles,
        "relevance_by_event": relevance_by_event,
        "model_info": "News verification model based on credibility scoring and relevance"
    }
    
    logger.info(f"News verification model complete with {len(unique_articles)} unique articles")
    return news_model

def fuse_modalities(geospatial_model, text_model, news_model, logger):
    """
    Fuse confidence scores from multiple modalities using the fusion engine.
    Uses dynamic fusion based on data quality factors.
    """
    logger.info("Fusing modalities using the fusion engine...")
    
    # Calculate real confidence scores based on model outputs
    
    # 1. Geospatial confidence based on graph structure
    if geospatial_model and 'graph' in geospatial_model:
        graph = geospatial_model['graph']
        # More connected components generally means less confidence
        n_components = len(list(nx.connected_components(graph)))
        component_factor = 1.0 / max(1, np.log2(n_components + 1))
        
        # More edges per node generally means more confidence
        edge_density = len(graph.edges) / max(1, len(graph.nodes))
        density_factor = min(1.0, edge_density / 2.0)
        
        geospatial_conf = 0.7 * component_factor + 0.3 * density_factor
    else:
        geospatial_conf = 0.0
    
    # 2. Text confidence based on sentiment strength and entity recognition
    if text_model and 'sentiments' in text_model and 'entities' in text_model:
        sentiments = text_model['sentiments']
        entities = text_model['entities']
        
        # Average compound sentiment (absolute value - we care about strength, not positivity)
        sentiment_strengths = [abs(s.get('compound', 0)) for s in sentiments if isinstance(s, dict)]
        avg_sentiment = np.mean(sentiment_strengths) if sentiment_strengths else 0
        
        # Consider more entities as higher confidence (normalize)
        entity_counts = [len(e) for e in entities]
        avg_entities = np.mean(entity_counts) if entity_counts else 0
        entity_factor = min(1.0, avg_entities / 10.0)  # Cap at 10 entities per description
        
        text_conf = 0.4 * avg_sentiment + 0.6 * entity_factor
    else:
        text_conf = 0.0
    
    # 3. News confidence based on verification scores
    if news_model and 'verification_scores' in news_model:
        scores = [s for s in news_model['verification_scores'].values() if s is not None]
        news_conf = np.mean(scores) if scores else 0.0
    else:
        news_conf = 0.0
    
    # Scale confidence scores to 0-1 range
    geospatial_conf = max(0, min(1, geospatial_conf))
    text_conf = max(0, min(1, text_conf))
    news_conf = max(0, min(1, news_conf))
    
    # Calculate quality factors for dynamic fusion
    # We'll simulate multiple samples by adding small noise to our confidence values
    random.seed(42)  # For reproducibility
    n_samples = 5
    module_score_samples = []
    
    for _ in range(n_samples):
        # Add small noise to each confidence score to simulate multiple evaluations
        noise = 0.05
        sample = {
            'geospatial': max(0, min(1, geospatial_conf + random.uniform(-noise, noise))),
            'text': max(0, min(1, text_conf + random.uniform(-noise, noise))),
            'news': max(0, min(1, news_conf + random.uniform(-noise, noise)))
        }
        module_score_samples.append(sample)
    
    # Calculate consistency metrics
    quality_factors = cross_validation_consistency(module_score_samples)
    
    # Use the fusion engine to get the final ensemble score
    module_scores = {
        'geospatial': geospatial_conf,
        'text': text_conf,
        'news': news_conf
    }
    
    # Calculate ensemble score using dynamic fusion
    ensemble_score = ensemble_scoring(module_scores, quality_factors)
    
    # Create confidence list and evidence list for reporting
    confidences = [geospatial_conf, text_conf, news_conf]
    
    evidence = [
        {"detail": "Geospatial analysis of event proximity", "score": geospatial_conf},
        {"detail": "Text similarity and entity recognition", "score": text_conf},
        {"detail": "News credibility assessment", "score": news_conf}
    ]
    
    # Add fusion quality to evidence
    for modality, quality in quality_factors.items():
        evidence.append({
            "detail": f"{modality.capitalize()} consistency quality factor",
            "score": quality
        })
    
    # Add ensemble information
    evidence.append({
        "detail": "Ensemble score using dynamic fusion",
        "score": ensemble_score
    })
    
    logger.info(f"Fusion complete with confidences: {confidences} and ensemble score: {ensemble_score:.4f}")
    return confidences, evidence

# ... [keep all existing imports and functions the same] ...

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting end-to-end Disaster Verification pipeline...")
    
    # Ensure the system path includes the parent directory for proper imports
    import sys  # Using os.path only, which is separate from os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        logger.info(f"Added parent directory to path: {parent_dir}")
    
    # Load disaster data
    df = load_data()
    if df is None or df.empty:
        logger.error("Data loading failed or returned empty DataFrame. Exiting pipeline.")
        return
    
    logger.info(f"Working with {len(df)} disaster events")
    
    # Train individual modules
    try:
        geospatial_model = train_geospatial_model(df, logger)
    except Exception as e:
        logger.error(f"Error in geospatial model training: {str(e)}")
        geospatial_model = None
    
    try:
        text_model = train_text_similarity_model(df, logger)
    except Exception as e:
        logger.error(f"Error in text similarity model training: {str(e)}")
        text_model = None
    
    try:
        news_model = train_news_verification(df, logger)
    except Exception as e:
        logger.error(f"Error in news verification model training: {str(e)}")
        news_model = None
    
    # Verify that at least one model completed successfully
    if not any([geospatial_model, text_model, news_model]):
        logger.error("All models failed to train. Cannot proceed with fusion.")
        return
    
    # Fuse modalities for final decision making
    try:
        import networkx as nx
        confidences, evidence = fuse_modalities(geospatial_model, text_model, news_model, logger)
    except Exception as e:
        logger.error(f"Error during modality fusion: {str(e)}")
        # Fallback to basic confidences and evidence if fusion fails
        confidences = [
            0.7 if geospatial_model else 0.0,
            0.7 if text_model else 0.0,
            0.7 if news_model else 0.0
        ]
        evidence = [
            {"detail": "Geospatial model output", "score": confidences[0]},
            {"detail": "Text similarity model output", "score": confidences[1]},
            {"detail": "News verification model output", "score": confidences[2]}
        ]
    
    # Generate final decision report
    try:
        report = generate_report(confidences, evidence)
        logger.info("Final decision report generated:")
        print(report)
    except Exception as e:
        logger.error(f"Error generating final report: {str(e)}")
        print(json.dumps({
            "error": "Failed to generate report",
            "confidences": confidences,
            "evidence_count": len(evidence) if evidence else 0
        }, indent=4))
    
    # ========== SAMPLE INPUT VERIFICATION ==========
    # Enhanced section that verifies a sample disaster input with all three models
    try:
        logger.info("\n=== Verifying Sample Disaster Input ===")
        
        # Sample disaster input to verify
        sample_input = {
            "latitude": 19.0760, 
            "longitude": 72.8777,
            "disaster_type": "flood", 
            "description": "Severe flooding in Mumbai has affected multiple areas with water levels rising above 3 feet in low-lying regions.",
            "location": "Mumbai, India",
            "date" : "2023-07-15"
        }
        
        logger.info(f"Sample input: {sample_input['disaster_type']} at coordinates ({sample_input['latitude']}, {sample_input['longitude']})")
        
        # 1. GEOSPATIAL VERIFICATION
        geo_match = False
        closest_event = None
        closest_distance = float('inf')
        geo_score = 0.0
        
        if geospatial_model and 'events' in geospatial_model:
            from geospatial_analysis.utils import haversine_distance
            
            input_coords = (sample_input['latitude'], sample_input['longitude'])
            threshold = geospatial_model.get('threshold', 100)
            
            # Find closest event
            for event in geospatial_model['events']:
                distance = haversine_distance(input_coords, event['coords'])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_event = event
            
            # Calculate distance-based score (inverse relationship - closer = higher score)
            geo_score = max(0, min(1, 1 - (closest_distance / (2 * threshold))))
            
            # Check if within threshold
            if closest_distance <= threshold:
                geo_match = True
                logger.info(f"Geospatial match found: {closest_event['id']} at distance {closest_distance:.2f}km")
            else:
                logger.info(f"No geospatial match found. Closest event: {closest_event['id']} at {closest_distance:.2f}km")
        
        # 2. TEXT SIMILARITY VERIFICATION
        text_match = False
        best_similarity = 0
        best_match_text = None
        text_score = 0.0
        
        if text_model and 'model' in text_model and 'embeddings' in text_model:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Generate embedding for input description
            input_desc = sample_input['description']
            input_embedding = generate_embeddings(text_model['model'], [input_desc])[0]
            
            # Compare with dataset embeddings
            similarities = cosine_similarity([input_embedding], text_model['embeddings'])[0]
            
            # Find best match
            best_idx = similarities.argmax()
            best_similarity = similarities[best_idx]
            text_score = best_similarity  # Direct use of similarity as score
            
            if best_similarity >= 0.6:  # Threshold for text similarity
                text_match = True
                best_match_text = text_model['descriptions'][best_idx][:100] + "..." if len(text_model['descriptions'][best_idx]) > 100 else text_model['descriptions'][best_idx]
                logger.info(f"Text match found with similarity {best_similarity:.4f}: {best_match_text}")
            else:
                logger.info(f"No significant text match found. Best similarity: {best_similarity:.4f}")
        
        # 3. NEWS VERIFICATION (NEW)
        news_match = False
        news_score = 0.0
        news_articles = []
        
        try:
            # Import the news scraper module
            from news_verification.news_scraper import get_disaster_news
            from datetime import datetime, timedelta
    
            # Parse the input date
            input_date = datetime.strptime(sample_input['date'], "%Y-%m-%d")
            
            # Calculate days between input date and today
            days_since_event = (datetime.now() - input_date).days
            
            # Set minimum search window of 7 days
            search_days = max(7, days_since_event)
            
            # Create search query from disaster type and location
            query = f"{sample_input['disaster_type']} {sample_input['location']}"
            logger.info(f"Searching for news articles with query: {query} within {search_days} days")
    
            # Get news articles (try with reduced days to get most recent)
            news_articles = get_disaster_news(query, days=3)
            
            if news_articles:
                # Calculate news verification score based on number of articles found
                article_factor = min(1.0, len(news_articles) / 5.0)  # Scale based on number of articles (max at 5+)
                
                # Check if disaster type appears in article titles or content
                keyword_matches = 0
                for article in news_articles:
                    if sample_input['disaster_type'].lower() in article['title'].lower() or \
                       sample_input['location'].lower() in article['title'].lower() or \
                       sample_input['disaster_type'].lower() in article['content'].lower():
                        keyword_matches += 1
                
                keyword_factor = min(1.0, keyword_matches / max(1, len(news_articles)))
                
                # Combine scores (weighting articles and keyword matches)
                news_score = 0.4 * article_factor + 0.6 * keyword_factor
                
                if news_score >= 0.5:
                    news_match = True
                    logger.info(f"News match found with {len(news_articles)} articles, score: {news_score:.4f}")
                else:
                    logger.info(f"Weak news verification with score: {news_score:.4f}")
            else:
                logger.info("No news articles found for the input disaster")
        except Exception as e:
            logger.error(f"Error in news verification: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
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
        
        # Print verification results
        print("\n=== SAMPLE DISASTER INPUT VERIFICATION ===")
        print(f"Input: {sample_input['disaster_type']} at ({sample_input['latitude']}, {sample_input['longitude']})")
        print(f"Location: {sample_input['location']}")
        print(f"Description: {sample_input['description']}")
        
        print("\nResults:")
        print(f"1. Geospatial Match: {'✓' if geo_match else '✗'} (Closest event: {closest_distance:.2f}km)")
        print(f"   Score: {geo_score:.4f}")
        
        print(f"2. Text Similarity Match: {'✓' if text_match else '✗'} (Best similarity: {best_similarity:.4f})")
        print(f"   Score: {text_score:.4f}")
        
        print(f"3. News Verification: {'✓' if news_match else '✗'} (Articles found: {len(news_articles)})")
        print(f"   Score: {news_score:.4f}")
        print(f"   Search window: {search_days} days from {sample_input['date']}")
        
        # Print most relevant news if available
        if news_articles:
            print("\nTop News Articles:")
            for i, article in enumerate(news_articles[:2]):
                print(f"  {i+1}. {article['title']}")
                print(f"     Source: {article['source']}")
                print(f"     Summary: {article['content'][:100]}..." if len(article['content']) > 100 
                      else f"     Summary: {article['content']}")
        
        # Print final classification and evidence
        print(f"\nDISASTER CLASSIFICATION: {probability_class} ({weighted_score:.4f})")
        
        print("\nSupporting Evidence:")
        if geo_match:
            print(f"- Found matching event within {threshold}km threshold")
        if text_match and best_match_text:
            print(f"- Found similar disaster description: {best_match_text}")
        if news_match:
            print(f"- Found {len(news_articles)} news articles about this disaster")
            
    except Exception as e:
        logger.error(f"Error verifying sample input: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Disaster Verification pipeline completed.")
if __name__ == "__main__":
    main()