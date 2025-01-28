import pandas as pd
from html import unescape
import pandas as pd
import re
import os
import nltk
import spacy
from nltk.corpus import stopwords

import data_io

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')  # Load the Spacy English model

# Cleaned file path(temporary)
cleaned_file_path = "/Cleaned_Data/"

def test():
    print("Test")
    return "Test"

def clean_html(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[\W_]+', ' ', text)
    return text

# Remove stopwords, punctuation, and non-alphanumeric characters. Perform lemmatization or stemming. Tokenize text data.
def text_cleaning_and_preprocessing(data):
    try:
        print("Text Cleaning and Preprocessing")

        data['review_body'] = data['review_body'].apply(clean_html)

        # Decode HTML entities and handle NaN values
        data['review_body'] = data['review_body'].apply(
            lambda x: unescape(str(x)) if isinstance(x, str) else ''
        )
        print("HTML entities decoded")

        # Remove non-alphanumeric characters and punctuation, preserving spaces
        data['review_body'] = data['review_body'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        print("Non-alphanumeric characters and punctuation removed")

        # Convert text to lowercase
        data['review_body'] = data['review_body'].str.lower()
        print("Text converted to lowercase")

        # Tokenize text
        data['review_body'] = data['review_body'].apply(
            lambda x: x.split() if isinstance(x, str) else []
        )
        print("Text tokenized")

        # Remove stopwords
        data['review_body'] = data['review_body'].apply(
            lambda x: [word for word in x if word not in stop_words]
        )
        print("Stopwords removed")

        # Perform lemmatization
        data['review_body'] = data['review_body'].apply(
            lambda x: [token.lemma_ for token in nlp(' '.join(x))] if x else []
        )
        print("Lemmatization done")

        # Join the tokens back into sentences
        data['review_body'] = data['review_body'].apply(lambda x: ' '.join(x))
        print("Tokens joined back into sentences")

        # Remove invalid rows
        invalid_rows = data[~data['review_body'].apply(lambda x: isinstance(x, str))]
        print("Invalid Rows:", invalid_rows)
        print("Invalid Rows Count:", len(invalid_rows))
        print("Invalid Rows Index:", invalid_rows.index)
        print("Invalid Rows Review Body:", invalid_rows['review_body'])

        return data
    except Exception as e:
        print("-" * 100)
        print(e)
        print(data['review_body'])

        # if failed to clean the data, return "None" to indicate the error
        return None


