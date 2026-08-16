import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
news_df = pd.read_csv("fake_or_real_news.csv")

# Features and labels
X = news_df["text"]
y = news_df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build ML pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("nbmodel", MultinomialNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)

print("Classification Report\n")
print(classification_report(y_test, predictions))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, predictions))

# Save trained model
with open("model.pickle", "wb") as file:
    pickle.dump(pipeline, file, protocol=pickle.HIGHEST_PROTOCOL)