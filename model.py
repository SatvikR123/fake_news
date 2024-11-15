import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Load the data
df = pd.read_csv('train.csv')   
print("Data loaded. Shape:", df.shape)

# Prepare data
X = df.drop('label', axis=1)
y = df['label']
voc_size = 5000
messages = X.copy()

# Check if 'title' column exists
if 'title' not in messages.columns:
    raise ValueError("Column 'title' not found in the dataset.")


messages['title'] = messages['title'].fillna('')

ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# One-hot encoding and padding
onehot_repr = [one_hot(words, voc_size) for words in corpus]
sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

# Creating the model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert data to numpy arrays
X_final = np.array(embedded_docs)
y_final = np.array(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Save the trained model
model.save("fake_news_model.h5")
print("Model saved as 'fake_news_model.h5'")
