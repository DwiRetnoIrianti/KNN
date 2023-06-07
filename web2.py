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

# Load the data
df = pd.read_csv("Data Ulasan Gojek.csv")

df.rename(columns={"userName": "Nama Pengguna", "score": "rating", "at": "tanggal", "content": "ulasan"}, inplace=True)


# mengganti label pada kolom rating
def replace_rating(x):
    if x <= 3:
        return 'Tidak Puas'
    else:
        return 'Puas'

df['rating'] = df['rating'].apply(replace_rating)

# Preprocessing functions
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


# Preprocess the reviews
df['processed_ulasan'] = df['ulasan'].apply(preprocess_text)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['processed_ulasan'])
Y = df['rating']

## Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediction function
def predict_satisfaction(review):
    processed_review = preprocess_text(review)
    transformed_review = tfidf.transform([processed_review])
    prediction = knn.predict(transformed_review)
    return prediction[0]

    positive_words = ["puas", "bagus", "cepat", "baik", "memuaskan"]
    negative_words = ["tidak puas", "buruk", "lama", "tidak baik", "mengecewakan"]

    positive_count = sum(1 for word in processed_review.split() if word in positive_words)
    negative_count = sum(1 for word in processed_review.split() if word in negative_words)

    if positive_count > negative_count:
        return "Puas"
    else:
        return "Tidak Puas"

# Streamlit web app
st.title('Klasifikasi Kepuasan Pelanggan Ojek Online (GOJEK)')
st.write('IMPLEMENTASI TEXT PADA KLASIFIKASI KEPUASAN PELANGGAN TRANSPORTASI ONLINE (OJEK ONLINE) MENGGUNAKAN METODE KNN')

# User input
user_input = st.text_area('Masukkan ulasan Anda:')

if st.button('Klasifikasi'):
    if user_input:
        # Make prediction using the trained KNN model
        prediction = predict_satisfaction(user_input)

        # Display the predicted label
        if prediction == 'Puas':
            st.write('Berdasarkan ulasan Anda, bahwa Anda merasa PUAS.')
        else:
            st.write('Berdasarkan ulasan Anda, bahwa Anda merasa TIDAK PUAS.')
    else:
        st.write('Masukkan ulasan Anda sebelum melakukan prediksi.')

