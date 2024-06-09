import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
CONFIG = {
    "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
    "local_path": "../models/second_sentiment_model",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"])

# Save model and tokenizer locally
tokenizer.save_pretrained(CONFIG["local_path"])
model.save_pretrained(CONFIG["local_path"])

logger.info(f"Model and tokenizer saved to {CONFIG['local_path']}")

# Move model to the appropriate device
model.to(CONFIG["device"])
