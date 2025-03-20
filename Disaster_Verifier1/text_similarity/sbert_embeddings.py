import argparse
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    """
    Load the SBERT model.
    """
    model = SentenceTransformer(model_name)
    return model

def generate_embeddings(model, sentences):
    """
    Generate embeddings for a list of sentences.
    """
    embeddings = model.encode(sentences)
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Generate SBERT embeddings for disaster descriptions.")
    parser.add_argument("--text", type=str, help="Input text for embedding generation", required=True)
    args = parser.parse_args()
    
    model = load_model()
    embedding = generate_embeddings(model, [args.text])
    
    print("Generated embedding:")
    print(embedding)

if __name__ == "__main__":
    main()