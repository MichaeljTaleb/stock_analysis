#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import re

#%%
# Load the first dataset
df1 = pd.read_csv('/Users/michaeltaleb/Documents/archive/good_data.csv', encoding='latin1')
df1.columns = ['Sentiment', 'Text']

# Load the second dataset
df2 = pd.read_csv('/Users/michaeltaleb/Documents/archive/all_data.csv', encoding='latin1')
df2.columns = ['Sentiment', 'Text']

# Load the third dataset, only selecting the needed columns
df3 = pd.read_csv('/Users/michaeltaleb/Documents/archive/general_data.csv', encoding='latin1', usecols=[0, 5])
df3.columns = ['Sentiment', 'Text']

# Map the sentiment values accordingly
df3['Sentiment'] = df3['Sentiment'].map({0: 'neutral', 4: 'positive'})


# Concatenate the datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

print("Number of columns in df1:", df1.shape[1])
print("Number of columns in df2:", df2.shape[1])
print("Number of columns in df3:", df3.shape[1])
print("Number of columns in combined df:", df.shape[1])
print(df.head())

#%%
# Encode the sentiment labels
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Preprocess the text data
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()) if isinstance(x, str) else x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)
print(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calculate and print percent accuracy
percent_accuracy = accuracy * 100
print(f'Percent Accuracy: {percent_accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Example new text
new_text = "net interest income totaled eur 159 mn compared to eur 156 mn a year earlier"
new_text_vec = vectorizer.transform([new_text])
new_text_prediction = model.predict(new_text_vec)
print(new_text_prediction[0])

# Sample texts
sample_texts = [
    "The company's innovative approach has significantly boosted its market share.",
    "Customer satisfaction has reached an all-time high due to the new service policies.",
    "The latest product launch was a huge success and exceeded all expectations.",
    "The recent layoffs have caused a lot of unrest among employees.",
    "The company's profits have plummeted following the new strategy.",
    "Customer complaints have surged due to the poor quality of the latest products.",
    "The company announced a new strategy during the annual meeting.",
    "There have been several changes in the management team recently.",
    "The quarterly report will be released next week."
]

def predict_sentiment(text):
    text_processed = re.sub(r'[^\w\s]', '', text.lower())
    text_vec = vectorizer.transform([text_processed])
    text_prediction = model.predict(text_vec)
    text_label = label_encoder.inverse_transform(text_prediction)
    return text_label[0]

predictions = {text: predict_sentiment(text) for text in sample_texts}

# Print the predictions
for text, sentiment in predictions.items():
    print(f'Text: {text}\nPredicted Sentiment: {sentiment}\n')

# Create a DataFrame to compare the actual and predicted sentiments
test_results = pd.DataFrame({
    'Text': X_test,
    'Actual Sentiment': label_encoder.inverse_transform(y_test),
    'Predicted Sentiment': label_encoder.inverse_transform(y_pred)
})

# Loop to input text and get sentiment analysis
while True:
    user_input = input("Enter text for sentiment analysis (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    sentiment = predict_sentiment(user_input)
    print(f'Predicted Sentiment: {sentiment}\n')

# Uncomment the following lines to print the test results
# for index, row in test_results.iterrows():
#     print(f'Text: {row["Text"]}\nActual Sentiment: {row["Actual Sentiment"]}\nPredicted Sentiment: {row["Predicted Sentiment"]}\n')
