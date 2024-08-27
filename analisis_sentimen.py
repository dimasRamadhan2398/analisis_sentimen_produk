import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.title('ANALISIS SENTIMEN PRODUK')
st.subheader(
    "Tools ini berguna untuk memprediksi sentimen produk berdasarkan komentar pengguna atau pembeli, apakah produk tersebut mendapatkan respon positif atau negatif"
)

dataset = pd.read_csv("reviews.csv")
corpus = []
for i in range(0, len(dataset)):
  review = re.sub('[^a-zA-Z]', ' ', dataset['reviews'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('indonesian')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

new_review_0 = st.text_input("Masukkan komentar atau review anda dibawah ini",
                             placeholder="masukkan komentar disini")
new_review = re.sub('[^a-zA-Z]', ' ', new_review_0)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('indonesian')
new_review = [
    ps.stem(word) for word in new_review if not word in set(all_stopwords)
]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = tfidf_vectorizer.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
st.button("Analisa Komen")
if not new_review_0:
  st.write(' ')
else:
  if new_y_pred == 1:
    st.write("Komen tersebut adalah komen yang :blue[POSITIF]")
  else:
    st.write("Komen tersebut adalah komen yang :red[NEGATIF]")
