from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Specify the local path where the model and tokenizer are saved
model_save_path = "../models/second_sentiment_model"

# Load the tokenizer and model from the local path
tokenizer = BertTokenizer.from_pretrained(model_save_path)
sentiment_model = BertForSequenceClassification.from_pretrained(model_save_path)
sentiment_model.eval()

# Function to make predictions
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    print(f"Tokenized input: {inputs}")  # Debug print for tokenized input

    # Perform inference with the model
    with torch.no_grad():  # Ensure no gradients are computed
        outputs = sentiment_model(**inputs)

    # Extract logits
    logits = outputs.logits
    print(f"Logits: {logits}")  # Debug print for logits

    # Get the predicted class
    prediction_index = torch.argmax(logits, dim=-1).item()
    print(f"Prediction index: {prediction_index}")  # Debug print for prediction index

    return prediction_index, logits

# Example usage
text = "Unemployment rates have increased and decreased profits."
prediction, logits = predict_sentiment(text)
print("Sentiment prediction index:", prediction)
print("Logits:", logits)

# You can map the prediction index to the label if needed
sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
sentiment = sentiment_labels.get(prediction, "unknown")
print("Sentiment label:", sentiment)
