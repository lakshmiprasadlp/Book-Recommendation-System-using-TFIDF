import streamlit as st
import pandas as pd
import pickle

# Load the models and data
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

data = pd.read_pickle('books_data.pkl')

def recommend_books(book_title, cosine_sim=cosine_sim):
    idx = data[data['title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[book_indices]

# Streamlit App Interface
st.title("Book Recommendation System")

# Input book title
book_title = st.text_input("Enter a book title:")

if book_title:
    try:
        recommended_books = recommend_books(book_title)
        st.write("### Top 10 Recommended Books:")
        for i, book in enumerate(recommended_books, start=1):
            st.write(f"{i}. {book}")
    except IndexError:
        st.write("Book not found. Please check the title and try again.")
