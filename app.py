import streamlit as st
import numpy as np
import re
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.optimizers import Adam
import pandas as pd
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Load model without compiling
model = load_model("fake_news_model.h5", compile=False)

# Recompile model with updated optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Constants
voc_size = 5000
sent_length = 20
ps = PorterStemmer()

# Preprocess input text
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    onehot_repr = [one_hot(review, voc_size)]
    return pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

# Function to fetch latest news
def fetch_latest_news(api_key, query="latest news", language="en", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&sortBy=publishedAt&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return [(article['title'], article['url']) for article in articles if article['title']]

# Streamlit app interface
st.title("Fake News Detection")
st.write("Enter the title and body of the news article to check if it is fake or real.")

# Input form
title_input = st.text_input("News Title", "")
body_input = st.text_area("News Body", "")

if st.button("Predict"):
    # Preprocess and predict
    if title_input:
        processed_text = preprocess_text(title_input)
        prediction = model.predict(processed_text)
        label = "Real" if prediction > 0.5 else "Fake"
        st.write(f"**Prediction:** {label}")
    else:
        st.warning("Please enter a title to check.")

# Section for fetching latest news and checking authenticity
st.write("\n---\n")
st.write("Fetch the latest news and check if it is fake or real using your API key.")
api_key = st.text_input("API Key", type="password")
if api_key and st.button("Fetch & Predict Latest News"):
    latest_articles = fetch_latest_news(api_key, page_size=10)  # Fetch top 10 articles
    if latest_articles:
        titles, urls = zip(*latest_articles)
        
        # Preprocess and predict for each article title
        processed_titles = [preprocess_text(title) for title in titles]
        predictions = [model.predict(title) for title in processed_titles]
        labels = [(pred > 0.5).astype("int32") for pred in predictions]
        
        # Display results for each of the 10 articles
        for title, url, label in zip(titles, urls, labels):
            prediction_text = "Fake" if label == 0 else "Real"
            st.write(f"**Title:** {title}")
            st.write(f"**URL:** [Link]({url})")
            st.write(f"**Prediction:** {prediction_text}")
            st.write("---")
    else:
        st.warning("No articles found or invalid API key.")
