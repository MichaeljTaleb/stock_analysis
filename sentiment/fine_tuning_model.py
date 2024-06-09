# # Credit: https://www.kdnuggets.com/how-to-fine-tune-bert-sentiment-analysis-hugging-face-transformers & Michael Taleb
# # email: michael.j.taleb@vanderbilt.edu
# # Description: this program is a component of the APIs/data_analysis file
# # it is written in a separate program so that it is easier to debug and write
# # it will be pasted into APIs/data_an alysis analyse_article function after completion and testing
# # the purpose of this program is to take in an article as a string parameter and return whether or not
# # it indicates to buy or sell a stock
# import torch
# # Using a virtual environment for dependencies:
# # active in terminal by "source venv/bin/activate" & "deactivate"
#
# from datasets import load_dataset
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np
#
# def tokenize_function(dataset):
#     return tokenizer(dataset['text'], padding="max_length", truncation=True, max_length=128)
#
# # Used to get the actual sentiments of all the sentences in a list to compare to model's predictions
# def get_sentiment(tokenized_datasets):
#     return [tokenized_datasets[i]["label"] for i in range(len(tokenized_datasets))]
#
# # Function used in the training parameter to find the accuracy of model
# def compute_metrics(p):
#     preds = np.argmax(p.predictions, axis=1)
#     labels = p.label_ids
#     accuracy = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#     }
#
# dataset = load_dataset("csv",
#                        data_files='/Users/michaeltaleb/Documents/archive/all_data.csv',
#                        encoding='ISO-8859-1',
#                        column_names=['label', 'text'])
#
# # Making sure labels are integers
# labels = {"negative": -1, "neutral": 0, "positive": 1}
# def convert_labels(row):
#     row['label'] = int(labels[row['label']])
#     return row
#
# # Convert labels to integers
# dataset = dataset.map(convert_labels, batched=False)
#
# print("before tokenizer")
# # Tokenizing
# tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# print("afer tokenizer")
#
#
# # Split dataset into 0.2 & 0.8, 80% to train the model and 20% to test
# train_testvalid = tokenized_datasets['train'].train_test_split(test_size=0.2)
# train_dataset = train_testvalid['train']
# test_dataset = train_testvalid['test']
#
# # Loads the data batch by batch into the BERT trainer in shuffled or normal order
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=4)
#
# print("before loading pretrained")
# # Loads in the untrained ternary (three-class / num_labels = 3) BERT model
# sentiment_model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
# print("after loading pretrained")
#
#
# training_args = TrainingArguments(
#     output_dir='./results',
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=3e-6,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
# )
#
# trainer = Trainer(
#     model=sentiment_model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics,
# )
#
# # Uses the train_dataset to train the AI model
# trainer.train()
#
# # Save the trained model and tokenizer
# model_save_path = "../models/first_sentiment_model"
# trainer.save_model(model_save_path)
# tokenizer.save_pretrained(model_save_path)
#
# # Uses the test_dataset to evaluate the model
# metrics = trainer.evaluate()
# print(metrics)
#
# def predict_sentiment(text):
#     # Tokenize the input text
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
#     print(f"Tokenized input: {inputs}")  # Debug print for tokenized input
#
#     # Perform inference with the model
#     with torch.no_grad():  # Ensure no gradients are computed
#         outputs = sentiment_model(**inputs)
#
#     # Extract logits
#     logits = outputs.logits
#     print(f"Logits: {logits}")  # Debug print for logits
#
#     # Get the predicted class
#     prediction_index = torch.argmax(logits, dim=-1).item()
#     print(f"Prediction index: {prediction_index}")  # Debug print for prediction index
#
#     return prediction_index, logits
#
#
# while True:
#
#     # get user input:
#     user_input = input("Input sentence to be analyzed")
#
#     # convert to int:
#     input = user_input
#
#     if input == "q":
#         print("You quit!")
#         break
#     else:
#         score, logits = predict_sentiment(user_input)
#         print(score)
#
# # Predicting
# # predictions = trainer.predict(test_dataset)
# # print(predictions)
#
# # Print actual sentiments from the test dataset
# # print(get_sentiment(test_dataset))
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
CONFIG = {
    "data_files": [
        "/Users/michaeltaleb/Documents/archive/good_data.csv",
        "/Users/michaeltaleb/Documents/archive/all_data.csv",
        "/Users/michaeltaleb/Documents/archive/general_data.csv"
    ],
    "encoding": "ISO-8859-1",
    "column_names": ["Sentiment", "Text"],
    "model_name": "ProsusAI/finbert",
    "output_dir": "./results",
    "batch_size": 16,  # Increase batch size
    "num_epochs": 3,
    "learning_rate": 3e-6,
    "weight_decay": 0.01,
    "subset_size": 50000  # Use a smaller subset for initial training
}


def tokenize_function(dataset):
    return tokenizer(dataset['Text'], padding="max_length", truncation=True, max_length=128)


def convert_labels(row):
    labels = {"negative": -1, "neutral": 0, "positive": 1}
    row['Sentiment'] = int(labels.get(row['Sentiment'], 0))  # Default to neutral if not found
    return row


def convert_general_labels(row):
    row['Sentiment'] = 0 if row['Sentiment'] == 0 else 1 if row['Sentiment'] == 4 else -1
    return row


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# Load and preprocess datasets
datasets = []
for file in CONFIG["data_files"]:
    if file == "/Users/michaeltaleb/Documents/archive/general_data.csv":
        # Load specific columns for the general dataset
        dataset = load_dataset("csv", data_files=file, encoding=CONFIG["encoding"],
                               column_names=["Sentiment", "col2", "col3", "col4", "col5", "Text"], usecols=[0, 5])
        dataset = dataset.map(convert_general_labels, batched=False)
    else:
        dataset = load_dataset("csv", data_files=file, encoding=CONFIG["encoding"], column_names=CONFIG["column_names"])
        dataset = dataset.map(convert_labels, batched=False)

    # Ensure Sentiment is of type int64
    dataset = dataset.map(lambda x: {"Sentiment": int(x["Sentiment"])})
    datasets.append(dataset['train'])

# Concatenate datasets and use a subset for initial training
combined_dataset = concatenate_datasets(datasets).shuffle(seed=42)
subset_dataset = combined_dataset.select(range(CONFIG["subset_size"]))

# Tokenize the combined dataset
logger.info("Before tokenizer")
tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
tokenized_dataset = subset_dataset.map(tokenize_function, batched=True)
logger.info("After tokenizer")

# Rename columns to fit the model requirements
tokenized_dataset = tokenized_dataset.rename_column("Sentiment", "labels")

# Split the dataset into training and testing sets
train_testvalid = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_testvalid['train']
test_dataset = train_testvalid['test']

# Create DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=CONFIG["batch_size"], num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], num_workers=4)

# Load the pre-trained model
logger.info("Before loading pretrained")
sentiment_model = BertForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=3)
logger.info("After loading pretrained")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    num_train_epochs=CONFIG["num_epochs"],
    weight_decay=CONFIG["weight_decay"],
    load_best_model_at_end=True,
)

# Trainer setup
trainer = Trainer(
    model=sentiment_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model_save_path = "../models/first_sentiment_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Evaluate the model
metrics = trainer.evaluate()
logger.info(metrics)


# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    logger.info(f"Tokenized input: {inputs}")

    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    logits = outputs.logits
    logger.info(f"Logits: {logits}")

    prediction_index = torch.argmax(logits, dim=-1).item()
    logger.info(f"Prediction index: {prediction_index}")

    return prediction_index, logits


# Loop to input text and get sentiment analysis
while True:
    user_input = input("Input sentence to be analyzed")

    if user_input.lower() == "q":
        logger.info("You quit!")
        break
    else:
        score, logits = predict_sentiment(user_input)
        logger.info(score)
