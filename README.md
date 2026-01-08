# DisastVerif 🌍🔍

A multi-modal AI-powered disaster verification system that combines geospatial analysis, natural language processing, and news verification to validate disaster reports and combat misinformation.

## 🎯 Overview

DisastVerif is an advanced machine learning system designed to verify the authenticity of disaster reports by analyzing multiple data sources: 
- **Geospatial Analysis**: Location-based verification using Graph-based GNN (GR-GNN) and traditional Haversine distance + KNN
- **Text Similarity**: Semantic analysis using SBERT embeddings, NER, sentiment analysis, and topic modeling
- **News Verification**: Credibility scoring through web scraping, source validation, and contextual relevance checks
- **Multi-Modal Fusion**: Ensemble scoring and weighted aggregation for final verification decision

## ✨ Features

- 🗺️ **Advanced Geospatial Verification**
  - Graph-based Recurrent Graph Neural Network (GR-GNN)
  - Traditional KNN with Haversine distance calculation
  - Geographic proximity analysis

- 📝 **Natural Language Processing**
  - SBERT (Sentence-BERT) embeddings for semantic similarity
  - Named Entity Recognition (NER)
  - Sentiment analysis
  - Topic modeling

- 📰 **News Verification Engine**
  - Real-time news scraping and collection
  - Source credibility scoring
  - Duplicate article detection
  - Contextual relevance flagging
  - NLP-based content filtering

- 🔄 **Multi-Modal Fusion**
  - Weighted aggregation of multiple data sources
  - Ensemble scoring algorithms
  - Dynamic fusion techniques
  - Cross-validation consistency checks

- 🎯 **Decision Making System**
  - Confidence scoring
  - Evidence-based reporting
  - Comprehensive verification logs

## 🏗️ Architecture

```
DisastVerif/
├── data/                      # Data storage
│   ├── raw/                   # Unprocessed input data
│   ├── processed/             # Preprocessed data
│   └── external/              # External reference datasets
│
├── geospatial_analysis/       # Location verification
│   ├── knn_haversine.py       # Traditional Haversine + KNN
│   ├── grgnn.py               # Graph-based GNN approach
│   └── utils.py               # Helper functions
│
├── text_similarity/           # Text analysis
│   ├── sbert_embeddings.py    # SBERT embeddings
│   └── text_analysis.py       # NER, sentiment, topic modeling
│
├── news_verification/         # News credibility
│   ├── news_scraper.py        # Web scraping
│   └── credibility_scoring.py # Source validation
│
├── fusion_engine/             # Data fusion
│   ���── fusion_logic.py        # Ensemble scoring
│   └── data_quality.py        # Quality checks
│
├── decision_module/           # Final decision
│   ├── decision_maker.py      # Verification logic
│   └── reporting.py           # Results formatting
│
├── saved_models/              # Trained models
├── api. py                     # Flask REST API
├── verification_service.py    # Core verification service
└── train_model.py             # Model training pipeline
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Eesh-487/DisastVerif.git
cd DisastVerif
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download required NLP models**
```bash
python -m spacy download en_core_web_sm
```

### Configuration

1. **Prepare your data**:  Place your disaster dataset in the `data/` directory
2. **Train models** (if needed):
```bash
python train_model.py
```

### Running the Application

#### Start the Flask API Server

```bash
python api.py
```

The API will be available at `http://localhost:5000`

#### Using the Verification Service

```python
from verification_service import verify_disaster, load_models

# Load pre-trained models
load_models()

# Verify a disaster report
disaster_data = {
    'latitude': 37.7749,
    'longitude': -122.4194,
    'description': 'Major earthquake reported in downtown area',
    'location': 'San Francisco, CA',
    'disaster_type': 'earthquake',
    'date': '2026-01-08'
}

result = verify_disaster(disaster_data)
print(f"Verification Probability: {result['probability']}")
print(f"Evidence: {result['evidence']}")
```

## 📡 API Endpoints

### POST `/api/disasters`

Verify and submit a disaster report. 

**Request Body:**
```json
{
    "latitude": 37.7749,
    "longitude": -122.4194,
    "description": "Major earthquake reported in downtown area",
    "location": "San Francisco, CA",
    "disaster_type": "earthquake",
    "date": "2026-01-08"
}
```

**Response:**
```json
{
    "success": true,
    "message":  "Disaster report verified and saved successfully",
    "verification":  {
        "probability": 0.87,
        "classification": "verified",
        "evidence": {
            "geospatial_score": 0.92,
            "text_similarity_score": 0.85,
            "news_verification_score": 0.84
        }
    }
}
```

## 🛠️ Technology Stack

- **Backend Framework**: Flask, Flask-CORS
- **Machine Learning**: scikit-learn, PyTorch, PyTorch Geometric
- **NLP**:  Transformers, Sentence-Transformers, NLTK, spaCy, Gensim
- **Geospatial**: Geopy, NetworkX
- **Web Scraping**: BeautifulSoup4, Requests
- **Data Processing**: Pandas, NumPy
- **Testing**: pytest

## 📊 Model Training

The system uses three main model types:

1. **Geospatial Model**: Graph-based GNN for location verification
2. **Text Similarity Model**: SBERT-based semantic analysis
3. **News Verification Model**: Multi-source credibility scoring

Models are automatically saved to `saved_models/` directory after training and can be loaded for inference.

## 🔧 Development

### Project Structure

- `api.py`: REST API endpoints and Flask application
- `verification_service.py`: Core verification logic and model loading
- `train_model.py`: Model training and data preprocessing pipeline
- `requirements.txt`: Python dependencies

### Logging

Logs are written to `disaster_verifier_service.log` for debugging and monitoring.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Eesh-487**
- GitHub: [@Eesh-487](https://github.com/Eesh-487)

## 🙏 Acknowledgments

- SBERT for semantic text embeddings
- PyTorch Geometric for graph neural networks
- Flask community for web framework support

## 📞 Support

For issues, questions, or contributions, please open an issue in the GitHub repository.

---

**Note**: This system is designed for research and educational purposes. Always verify critical disaster information through official channels and emergency services. 
