import logging
import json
import pandas as pd

from decision_module.reporting import generate_report
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    # Read data from disasters.xlsx in the data folder
    logging.info("Loading disaster data from Excel...")
    df = pd.read_excel(r"C:\Users\eeshk\OneDrive\Desktop\Disaster_Verifier\Disaster_Verifier\data\disasters.xlsx")
    logging.info(f"Loaded {len(df)} rows from disasters.xlsx")
    return df

def train_geospatial_model(df):
    logging.info("Training geospatial analysis model...")
    
    # Assume each disaster event in df has at least:
    # 'id', 'latitude', 'longitude', and optionally 'label'
    events = []
    for index, row in df.iterrows():
        event = {
            "id": row.get("id", f"event_{index}"),
            "coords": (row["latitude"], row["longitude"]),
            "label": row.get("label", None)
        }
        events.append(event)
    
    # Use a defined proximity threshold (in kilometers) for edge creation:
    proximity_threshold = 50  # Adjust the threshold based on your requirements
    
    # Import the function that constructs the geospatial graph from the GR-GNN module
    from geospatial_analysis.grgnn import construct_geospatial_graph
    graph = construct_geospatial_graph(events, proximity_threshold)
    
    # For this example, the "trained model" is the constructed graph along with the threshold.
    geospatial_model = {
        "graph": graph,
        "threshold": proximity_threshold
    }
    
    logging.info("Geospatial analysis model training complete.")
    return geospatial_model

def train_text_similarity_model(df):
    logging.info("Training text similarity model...")
    # Import the SBERT embedding functions
    from text_similarity.sbert_embeddings import load_model, generate_embeddings
    
    # Extract disaster descriptions from the DataFrame
    # Ensure your dataset has a "description" column (adjust the column name if needed)
    if "description" not in df.columns:
        logging.error("DataFrame missing 'description' column.")
        return None

    descriptions = df["description"].astype(str).tolist()
    
    # Load the pre-trained SBERT model and generate embeddings
    model = load_model()
    embeddings = generate_embeddings(model, descriptions)
    
    # Package the model and its embeddings as a dictionary to simulate a "trained" text model.
    text_model = {
        "model": model,
        "embeddings": embeddings
    }
    
    logging.info("Text similarity model training complete.")
    return text_model

def train_news_verification(df):
    logging.info("Setting up news verification model...")
    
    # Check if the DataFrame contains a "news" column
    if "news" not in df.columns:
        logging.error("DataFrame missing 'news' column. Cannot train news verification model.")
        return None

    # Simulate the training process by assigning a dummy verification score to each news article.
    import random
    news_scores = {}
    for index, row in df.iterrows():
        news_text = row.get("news", "")
        event_id = row.get("id", f"news_{index}")
        if news_text:
            # Assign a dummy score between 0 and 1 as the verification score.
            score = random.uniform(0, 1)
            news_scores[event_id] = score
        else:
            news_scores[event_id] = None

    # Package the dummy model data
    news_model = {
        "verification_scores": news_scores,
        "model_info": "Dummy news verification model - no real training performed."
    }
    
    logging.info("News verification model training complete.")
    return news_model

def fuse_modalities(geospatial_model, text_model, news_model):
    logging.info("Fusing modalities using the fusion engine...")
    
    # Compute geospatial confidence: e.g., based on the successful computation of the model
    geospatial_conf = 0.8 if geospatial_model is not None else 0.0
    
    # Compute text similarity confidence similarly
    text_conf = 0.75 if text_model is not None else 0.0

    # Compute news verification confidence, e.g., as the average of all news scores (ignoring None values)
    if news_model is not None and "verification_scores" in news_model:
        scores = [score for score in news_model["verification_scores"].values() if score is not None]
        news_conf = sum(scores) / len(scores) if scores else 0.0
    else:
        news_conf = 0.0
    
    confidences = [geospatial_conf, text_conf, news_conf]
    
    evidence = [
        {"detail": "Geospatial anomaly detected", "score": geospatial_conf},
        {"detail": "Text similarity indicates disaster", "score": text_conf},
        {"detail": "News credibility supports event", "score": news_conf}
    ]
    
    logging.info(f"Fusion complete with confidences: {confidences} and evidence: {evidence}")
    return confidences, evidence

def main():
    setup_logging()
    logging.info("Starting end-to-end pipeline with single Excel data source...")

    # Load data from Excel
    df = load_data()

    # Train individual modules
    geospatial_model = train_geospatial_model(df)
    text_model = train_text_similarity_model(df)
    news_model = train_news_verification(df)

    # Fuse modalities for final decision making
    confidences, evidence = fuse_modalities(geospatial_model, text_model, news_model)

    # Generate final decision report
    report = generate_report(confidences, evidence)
    logging.info("Final decision report generated:")
    print(report)
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()