import os
import pandas as pd
import re
import requests
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from datasets import load_dataset

# Function to download the dataset from GitHub
@st.cache_resource
def download_dataset_from_github():
    github_url = "https://raw.githubusercontent.com/Mansur-Mahdee/Know_your_movies/refs/heads/main/data/movie_dataset.csv"
    dataset_path = "/tmp/movie_dataset.csv"
    
    if os.path.exists(dataset_path):
        st.write("Dataset already exists, skipping download.")
        return dataset_path

    try:
        response = requests.get(github_url)
        response.raise_for_status()
        with open(dataset_path, 'wb') as f:
            f.write(response.content)
        st.write(f"Downloaded dataset from GitHub to {dataset_path}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error during dataset download: {e}")
        return None
    
    return dataset_path


@st.cache_data
def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df_relevant = df[['title', 'director', 'cast', 'overview', 'budget', 'genres']]
    df_relevant.rename(columns={'overview': 'plot'}, inplace=True)
    return df_relevant

# Function to generate an answer based on the query
def generate_answer(query, movie_data):
    query = query.strip().lower()
    query = query.rstrip("?")

    def normalize_title(title):
        return re.sub(r'[^\w\s]', '', title.strip().lower())  # Remove non-alphanumeric characters

    # Normalize movie titles in the dataset
    movie_data['normalized_title'] = movie_data['title'].apply(normalize_title)

    response = "Movie not found."

    if "directed" in query:
        movie_title = query.split('directed')[1].strip()
        normalized_movie_title = normalize_title(movie_title)
        movie_info = movie_data[movie_data['normalized_title'] == normalized_movie_title]
        if not movie_info.empty:
            response = movie_info['director'].values[0]

    elif "plot" in query or "summary" in query:
        movie_title = query.split('plot of')[1].strip() if 'plot of' in query else query.split('plot for')[1].strip()
        normalized_movie_title = normalize_title(movie_title)
        movie_info = movie_data[movie_data['normalized_title'] == normalized_movie_title]
        if not movie_info.empty:
            response = movie_info['plot'].values[0]

    elif "starred in" in query or "starring" in query or "casts for" in query:
        movie_title = query.split('starred in')[1].strip() if 'starred in' in query else query.split('starring')[1].strip() if 'starring' in query else query.split('casts for')[1].strip()
        normalized_movie_title = normalize_title(movie_title)
        movie_info = movie_data[movie_data['normalized_title'] == normalized_movie_title]
        if not movie_info.empty:
            response = movie_info['cast'].values[0]

    elif "budget" in query:
        movie_title = query.split('budget for')[1].strip()
        normalized_movie_title = normalize_title(movie_title)
        movie_info = movie_data[movie_data['normalized_title'] == normalized_movie_title]
        if not movie_info.empty:
            response = movie_info['budget'].values[0]

    elif "tell me about" in query:
        movie_title = query.split('tell me about')[1].strip()
        normalized_movie_title = normalize_title(movie_title)
        movie_info = movie_data[movie_data['normalized_title'] == normalized_movie_title]
        if not movie_info.empty:
            response = movie_info[['title', 'plot', 'director', 'cast', 'genres', 'budget']].to_dict(orient='records')[0]

    return response

# Streamlit app
def main():
    st.title("Know Your Movies")

    # Download the dataset from GitHub
    dataset_path = download_dataset_from_github()

    if dataset_path:
        # Load the dataset
        df_relevant = load_dataset(dataset_path)

        # User input for the query
        user_query = st.text_input("Enter your movie query (type 'exit' to quit):")
        st.write("Suggested question: Who directed Avatar? What are the casts for Avenger? What is the plot for The Batman? You can also ask about a movie's budget, plot or a summary. Try it!")

        if user_query.lower() != 'exit':
            answer = generate_answer(user_query, df_relevant)
            st.write(f"Answer: {answer}")
        else:
            st.write("Exiting the application.")

if __name__ == "__main__":
    main()
