import numpy as np
import re
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
import pandas as pd  # Make sure to import pandas

# Load the trained model
model = load_model("fake_news_model.h5")
print("Model loaded successfully.")

# Constants
voc_size = 5000
sent_length = 20
ps = PorterStemmer()

# Function to fetch the latest news
def fetch_latest_news(api_key, query="latest news", language="en", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&sortBy=publishedAt&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return [(article['title'], article['url']) for article in articles if article['title']]


# Preprocess news titles
def preprocess_news_titles(titles):
    corpus = []
    for title in titles:
        review = re.sub('[^a-zA-Z]', ' ', title)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    onehot_repr = [one_hot(words, voc_size) for words in corpus]
    return pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

# Example of usage
api_key = "3c4912c2d71643ad9dd4db6700a648dd"  # Replace with your actual API key
latest_articles = fetch_latest_news(api_key)
titles, urls = zip(*latest_articles)  # Unpack titles and URLs

processed_titles = preprocess_news_titles(titles)

# Predict and create a DataFrame for results
predictions = model.predict(processed_titles)
labels = (predictions > 0.5).astype("int32")


print("\nLatest News Predictions:\n")
for title, url, label in zip(titles, urls, labels.flatten()):
    prediction_text = "Fake" if label == 0 else "Real"
    print(f"Title: {title}")
    print(f"URL: {url}")
    print(f"Prediction: {prediction_text}\n")
