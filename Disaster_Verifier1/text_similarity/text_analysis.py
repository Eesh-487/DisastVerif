import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Ensure that the VADER lexicon is downloaded
nltk.download("vader_lexicon", quiet=True)

def perform_ner(text, model="en_core_web_sm"):
    """
    Perform Named Entity Recognition (NER) on the input text using spaCy.
    """
    nlp = spacy.load(model)
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def analyze_sentiment(text):
    """
    Analyze the sentiment of the input text using VADER.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def topic_modeling(texts, n_topics=2, n_top_words=5):
    """
    Perform topic modeling using Latent Dirichlet Allocation (LDA).
    
    Parameters:
        texts (list): List of text documents.
        n_topics (int): Number of topics to extract.
        n_top_words (int): Number of words per topic.
    
    Returns:
        dict: Topics with their top words.
    """
    # Create a document-term matrix using CountVectorizer.
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    topics = {}
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic_weights in enumerate(lda.components_):
        top_feature_indices = topic_weights.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_feature_indices]
        topics[f"Topic {idx+1}"] = top_words
    return topics

if __name__ == "__main__":
    sample_text = (
        "Severe flooding has caused significant disruption in the city, "
        "with emergency services working tirelessly to rescue residents. "
        "Heavy rains and overflowing rivers have led to unprecedented damage."
    )
    
    print("=== Named Entity Recognition (NER) ===")
    entities = perform_ner(sample_text)
    print(entities)
    
    print("\n=== Sentiment Analysis ===")
    sentiment = analyze_sentiment(sample_text)
    print(sentiment)
    
    print("\n=== Topic Modeling ===")
    topics = topic_modeling([sample_text], n_topics=1, n_top_words=5)
    print(topics)