
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("c:\Avneesh\DSA\project\Project-5\project\data\IMBD_Data.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    return " ".join(words)

df["clean_story"] = df["Story"].apply(clean_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["clean_story"])

def recommend_movies(storyline):
    cleaned = clean_text(storyline)
    vector = vectorizer.transform([cleaned])
    similarity = cosine_similarity(vector, tfidf_matrix)
    scores = similarity[0]
    top_indices = scores.argsort()[-5:][::-1]
    return df.iloc[top_indices][["movie_name","storyline"]]

st.title("Movie Recommendation System")

story = st.text_area("Enter Movie Storyline")

if st.button("Recommend"):
    results = recommend_movies(story)
    for _, row in results.iterrows():
        st.subheader(row["movie_name"])
        st.write(row["storyline"])
