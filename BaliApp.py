
import pandas as pd
import streamlit as st
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer
import pickle as pkl
from tqdm import tqdm
import re
from summarizer import Summarizer



# Read data
bali_reviews = pd.read_csv('Bali.csv', header=0)

# Define stop words
stopwords = list(STOP_WORDS) + ['room','hotel','rooms','hotels']

# Define functions
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

# Choosing summarizer model
model = Summarizer()

def summarized_review(data):
    data = data.values[0]
    return model(data, num_sentences=3)

class BaliRecs:

    def __init__(self):
        # Define embedder
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def clean_data(self):
        # Aggregate all reviews for each hotel
        aggregate_reviews = bali_reviews.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(
            ''.join).reset_index(name='review_body')

        # Review summary
        aggregate_summary = aggregate_reviews.copy()
        aggregate_summary['summary'] = aggregate_summary[["review_body"]].apply(summarized_review, axis=1)

        # Retain only alpha numeric characters
        aggregate_reviews['review_body'] = aggregate_reviews['review_body'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

        # Change to lowercase
        aggregate_reviews['review_body'] = aggregate_reviews['review_body'].apply(lambda x: lower_case(x))

        # Remove stop words
        aggregate_reviews['review_body'] = aggregate_reviews['review_body'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))

        # Retain the parsed review body in the summary df
        aggregate_summary['review_body'] = aggregate_reviews['review_body']

        df_sentences = aggregate_reviews.set_index("review_body")
        df_sentences = df_sentences["hotelName"].to_dict()
        df_sentences_list = list(df_sentences.keys())

        # Embeddings
        corpus = [str(d) for d in tqdm(df_sentences_list)]
        corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)

        # Dump to pickle file to use later for prediction
        with open("corpus.pkl", "wb") as file1:
            pkl.dump(corpus, file1)

        with open("corpus_embeddings.pkl", "wb") as file2:
            pkl.dump(corpus_embeddings, file2)

        with open("aggregate_reviews.pkl", "wb") as file3:
            pkl.dump(aggregate_reviews, file3)

        with open("aggregate_summary.pkl", "wb") as file4:
            pkl.dump(aggregate_summary, file4)

        return aggregate_summary, aggregate_reviews, corpus, corpus_embeddings


    def construct_app(self):
        aggregate_summary, aggregate_reviews, corpus, corpus_embeddings = self.clean_data()


        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Hotel Recommender System </p>',
            unsafe_allow_html=True
        )

        # Print summarized text
        st.markdown("Aggregated reviews")
        st.dataframe(aggregate_reviews)
        st.markdown("Aggregated summary")
        st.dataframe(aggregate_summary)
        st.markdown("Corpus")
        st.write(corpus)
        st.markdown("Corpus Embeddings")
        st.write(corpus_embeddings)

        return self

br = BaliRecs()
br.construct_app()