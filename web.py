import streamlit as st
import pandas as pd
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords')

st.title("Analisis Sentimen menggunakan KNN")

# Muat data
df = pd.read_csv("Data Ulasan Gojek.csv")

# Pra-pemrosesan
df.rename(columns={"userName": "Nama Pengguna", "score": "rating", "at": "tanggal", "content": "ulasan"}, inplace=True)


def replace_rating(x):
    if x <= 2:
        return 'Tidak Puas'
    else:
        return 'Puas'


df['rating'] = df['rating'].apply(replace_rating)

df['ulasan'] = df['ulasan'].apply(lambda x: x.lower())
df['ulasan'] = df['ulasan'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation + string.digits), '', x))
df['ulasan'] = df['ulasan'].apply(lambda x: x.split())

stop_words = set(stopwords.words('indonesian'))
df['ulasan'] = df['ulasan'].apply(lambda x: [word for word in x if word not in stop_words])

stemmer = PorterStemmer()
df['ulasan'] = df['ulasan'].apply(lambda x: [stemmer.stem(word) for word in x])

df['ulasan'] = df['ulasan'].apply(lambda x: ' '.join(x))

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['ulasan'])
Y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


def preprocess_text(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation + string.digits), '', text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = [stemmer.stem(word) for word in text]
    text = ' '.join(text)
    return text


def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vector = tfidf.transform([processed_text])
    prediction = knn.predict(text_vector)
    return prediction[0]


# Antarmuka Web
input_text = st.text_input("Masukkan ulasan Anda:", "")
if st.button("Analisis"):
    sentiment = predict_sentiment(input_text)
    st.write("Sentimen:", sentiment)
