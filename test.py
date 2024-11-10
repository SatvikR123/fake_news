import requests
import pandas as pd
def fetch_trusted_sources(api_key, query='latest news', language='en', page_size=5):
    url = f"https://newsapi.org/v2/top-headlines?sources=bbc-news&apiKey={api_key}"  # Replace with your sources
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return [article['title'] + '. ' + article['content'] for article in articles if article['content']]
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def verify_article(article, trusted_articles):
    articles = trusted_articles + [article]
    tfidf = TfidfVectorizer().fit_transform(articles)
    vectors = tfidf.toarray()
    csim = cosine_similarity(vectors)
    return csim[-1][:-1]  # Similarity scores with trusted articles

import numpy as np

def classify_article(article, trusted_articles, threshold=0.5):
    if not trusted_articles:  # Check if trusted_articles is empty
        return "No trusted articles available for verification."

    similarity_scores = verify_article(article, trusted_articles)
    
    # Check if similarity_scores is a numpy array and if it's empty
    if isinstance(similarity_scores, np.ndarray) and similarity_scores.size == 0:
        return "No similarity scores available."

    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    classification = "Likely Real" if avg_similarity > threshold else "Likely Fake"

    # Print the article being classified and the classification result
    print(f"Article: {article}")
    print(f"Average Similarity Score: {avg_similarity:.2f}")
    return classification


# Fetch trusted news articles
api_key = "3c4912c2d71643ad9dd4db6700a648dd"  # Replace with your actual API key
trusted_articles = fetch_trusted_sources(api_key)

# Preprocess the trusted articles
trusted_articles = [preprocess_text(article) for article in trusted_articles]

# Classify a new article
new_article = "Input your news article text here."
processed_article = preprocess_text(new_article)
classification = classify_article(processed_article, trusted_articles)

print(f"Classification of the article: {classification}")
