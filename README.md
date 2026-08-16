# Fake News Detection

A Flask-based web application that classifies a news article as **Fake** or **Real** by extracting the content from a news article URL and analyzing it using Machine Learning and Natural Language Processing (NLP). The application uses a trained TF-IDF and Multinomial Naive Bayes model for prediction.

## Features

- Predicts whether a news article is Fake or Real
- Accepts a news article URL as input
- Automatically extracts article content using Newspaper3k
- Uses TF-IDF and Multinomial Naive Bayes for classification
- Simple and user-friendly web interface

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Newspaper3k
- HTML
- CSS

## Installation

1. Download or clone this repository.
2. Open the project folder.
3. Install the required dependencies by running `pip install -r requirements.txt`.

## Run the Application

Run `python app.py`.

Then open your browser and visit `http://127.0.0.1:5000`.

## Train the Model

To retrain the model using the dataset, run `python fake_news_detection.py`.