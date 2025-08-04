import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load JSON data
with open('afrida.json', 'r') as file:
    data = json.load(file)

# Prepare data for training
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Convert to DataFrame
df = pd.DataFrame({"pattern": patterns, "tag": tags})

# Encode labels
label_encoder = LabelEncoder()
df['tag_encoded'] = label_encoder.fit_transform(df['tag'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['pattern'], df['tag_encoded'], test_size=0.2, random_state=42
)

# Vectorize text patterns
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model (SVM)
model = LinearSVC(max_iter=1000, random_state=42)  # Ensuring 100 iterations for the SVM model
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def get_response(user_input):
    # Vectorize user input
    user_input_tfidf = vectorizer.transform([user_input])

    # Predict intent
    intent_encoded = model.predict(user_input_tfidf)[0]
    intent = label_encoder.inverse_transform([intent_encoded])[0]

    # Fetch a random response for the predicted intent
    for intent_data in data['intents']:
        if intent_data['tag'] == intent:
            response = random.choice(intent_data['responses'])
            return response

user_input = "who is the hod of cse?"
response = get_response(user_input)
print("Bot:", response)
