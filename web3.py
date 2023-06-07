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

df = pd.read_csv("Data Ulasan Gojek.csv")
df.rename(columns={"userName": "Nama Pengguna", "score": "rating", "at": "tanggal", "content": "ulasan"}, inplace=True)

# mengganti label pada kolom rating
def replace_rating(x):
    if x <= 3:
        return 'Tidak Puas'
    else:
        return 'Puas'

df['rating'] = df['rating'].apply(replace_rating)

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation + string.digits), '', text)
    text = text.split()
    stop_words = set(stopwords.words('indonesian'))
    text = [word for word in text if word not in stop_words]
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    text = ' '.join(text)
    return text

df['processed_ulasan'] = df['ulasan'].apply(preprocess_text)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['processed_ulasan'])
Y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Kata-kata positif
positive_words = [ 'senang', 'baik', 'bagus', 'mantap']

# Kata-kata negatif
negative_words = ['tidak puas', 'kecewa', 'buruk', 'jelek', 'mengecewakan']


# Fungsi untuk melakukan klasifikasi berdasarkan kata-kata positif dan negatif
def classify_sentiment(review):
    processed_review = preprocess_text(review)
    words = processed_review.split()
    positive_count = sum(word in positive_words for word in words)
    negative_count = sum(word in negative_words for word in words)

    if positive_count > negative_count:
        return 'Puas'
    elif positive_count < negative_count:
        return 'Tidak Puas'
    else:
        return 'Netral'


# Streamlit web app
st.title('Klasifikasi Kepuasan Pelanggan Ojek Online (GOJEK)')
st.write(
    'IMPLEMENTASI TEXT PADA KLASIFIKASI KEPUASAN PELANGGAN TRANSPORTASI ONLINE (OJEK ONLINE) MENGGUNAKAN METODE KNN')

# User input
user_input = st.text_area('Masukkan ulasan Anda:')

if st.button('Klasifikasi'):
    if user_input:
        # Make sentiment classification using positive and negative words
        sentiment = classify_sentiment(user_input)

        # Display the sentiment label
        if sentiment == 'Puas':
            st.write('Berdasarkan ulasan Anda, bahwa Anda merasa PUAS.')
        elif sentiment == 'Tidak Puas':
            st.write('Berdasarkan ulasan Anda, bahwa Anda merasa TIDAK PUAS.')
        else:
            st.write('Berdasarkan ulasan Anda, tidak dapat diklasifikasikan sebagai Puas atau Tidak Puas.')
    else:
        st.write('Masukkan ulasan Anda sebelum melakukan prediksi.')
