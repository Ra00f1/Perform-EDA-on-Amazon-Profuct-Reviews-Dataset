import re
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.lang.en import English
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import vaderSentiment
import textblob
import sklearn
from textblob import TextBlob
from wordcloud import WordCloud
import gensim
import streamlit as st
import plotly
from collections import Counter
import kagglehub
import zipfile
import os
from html import unescape
from rake_nltk import Rake
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import data_io
import matplotlib
import Visualization
from sklearn.feature_extraction.text import CountVectorizer

matplotlib.use('TkAgg')  # Or use 'Agg' for non-interactive environments

# pandas disable low memory reading
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

Test_Path = "Data/"

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')  # Load the Spacy English model


def explor_data():
    print("Reading files")
    for file in os.listdir(Test_Path):
        print("_" * 100)
        print(file)
        file_path = Test_Path + file
        # Read and display the first 5 rows of the dataset
        df = pd.read_csv(file_path, sep="\t", on_bad_lines='warn', low_memory=False)


def data_info(data):
    try:
        print("_" * 100)
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.columns)
        print(data.shape)
        print(data.isnull().sum())
        print(data.dtypes)
        print(data.nunique())
        print("_" * 100)
    except Exception as e:
        print(e)


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


def process_in_chunks(file_name, file_path=Test_Path, chunk_size=10000, max_chunks=150, chunk_num=0):
    """
    Process a large dataset in chunks and save the processed chunks to new files.
    Uses the text_cleaning_and_preprocessing() function to clean the text data.

    :param file_name: Name of the file to process
    :param file_path: Path to the file
    :param chunk_size: Size of each chunk to read
    :param max_chunks: Maximum number of chunks to process
    :param chunk_num: Starting chunk number(optional)
    :return: DataFrame
    """
    chunks = []
    file_path = file_path + file_name

    try:
        for chunk in pd.read_csv(file_path, sep="\t", on_bad_lines='warn', low_memory=False, chunksize=chunk_size):
            if chunk_num >= max_chunks:
                print("Max chunks reached.")
                break

            print(f"Processing chunk: {chunk_num + 1}")
            chunk = text_cleaning_and_preprocessing(chunk)

            if chunk is not None:
                # Save the processed chunk to a new file
                chunk.to_csv(
                    Test_Path + f"Chunks/processed_{file_name}_{chunk_num + 1}.csv",
                    index=False
                )
                chunks.append(chunk)
            else:
                print(f"Chunk {chunk_num + 1} had a problem. Skipping...")

            chunk_num += 1

    except pd.errors.EmptyDataError:
        print("Reached the end of the file or encountered an empty chunk.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def calculate_sentiment(text):
    print("Calculating Sentiment")
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


# This function runs all the necessary steps to download the dataset from Kaggle and extract the contents and save
# them etc.
def one_and_done():
    data = data_io.read_tsv("amazon_reviews_us_Books_v1_02.tsv", Test_Path)

    data = text_cleaning_and_preprocessing(data)

    data['polarity'], data['subjectivity'] = zip(*data['review_body'].apply(calculate_sentiment))

    data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)

    Visualization.correlation_charts(Test_Path + "cleaned_data_test_html.csv")

    Visualization.word_cloud(data)

    Visualization.RAKE(Test_Path + "cleaned_data_test_html.csv")

    Visualization.LDA_topic_modeling(data['review_body'].tolist())

    Visualization.summarize_insights_and_visualize(data, num_keywords=50)

    print("Done")


# do all the processing in chunks for all the files in the directory
def process_all_files():
    for file in os.listdir(Test_Path):
        print("_" * 100)
        print(file)
        file_path = Test_Path + file

        data = process_in_chunks(file, Test_Path, chunk_size=10000, max_chunks=150, chunk_num=0)

        data['polarity'], data['subjectivity'] = zip(*data['review_body'].apply(calculate_sentiment))

        # Ssave the file by adding "cleaned" to the file name
        data.to_csv(Test_Path + "cleaned_" + file, index=False)

        # Delete the original file to save space
        os.remove(file_path)

        Visualization.correlation_charts(file_path)

        Visualization.word_cloud(data)

        Visualization.RAKE(file_path)

        Visualization.LDA_topic_modeling(data['review_body'].tolist())

        Visualization.summarize_insights_and_visualize(data, num_keywords=50)

        print("Done")


# TODO: Write a function about reading the files


if __name__ == '__main__':
    # download_dataset()
    # data_io.unzip_data()
    # explor_data()

    # Initial data ---------------------------------------------------------------------------------------------------

    # data = data_io.read_tsv("amazon_reviews_us_Books_v1_02.tsv", Test_Path, max_rows=1000)
#
    # # data_info(data)
    # data = text_cleaning_and_preprocessing(data)
    # # data_info(data)
    # data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)
#
    # # Process data in chunks when reading large files
    # data = data_io.process_in_chunks("amazon_reviews_us_Books_v1_02.tsv", Test_Path, chunk_size=10000, max_chunks=400,
    #                           chunk_num=35)
    # data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)
    # # Analyze the sentiment of reviews and add polarity and subjectivity scores to the DataFrame
#
    # data['polarity'], data['subjectivity'] = zip(*data['review_body'].apply(calculate_sentiment))
#
    # # Save the cleaned data
    # data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)
#
    # Analyze sentiment of reviews and add the scores to the DataFrame
    # analyzer = SentimentIntensityAnalyzer()
    # data['sentiment'] = data['review_body'].apply(lambda x: analyzer.polarity_scores(x))
    # data['neg'] = data['sentiment'].apply(lambda x: x['neg'])
    # data['neu'] = data['sentiment'].apply(lambda x: x['neu'])
    # data['pos'] = data['sentiment'].apply(lambda x: x['pos'])
    # data['compound'] = data['sentiment'].apply(lambda x: x['compound'])
#
    # # Save the cleaned data
    # data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)

    # ---------------------------------------------------------------------------------------------------------------

    file_name = "cleaned_data_test_html.csv"

    file_path = Test_Path + file_name
    text_column = ["star_rating"]

    data = data_io.read_csv(file_path, columns=text_column)

    # make a pie chart of positive and negative and neutral reviews count

    positive = data[data['star_rating'] >= 4]
    negative = data[data['star_rating'] <= 2]
    neutral = data[(data['star_rating'] == 3)]

    positive_count = positive.shape[0]
    negative_count = negative.shape[0]
    neutral_count = neutral.shape[0]

    labels = ['Positive', 'Negative', 'Neutral']

    sizes = [positive_count, negative_count, neutral_count]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')

    plt.show()

    print("Done")
