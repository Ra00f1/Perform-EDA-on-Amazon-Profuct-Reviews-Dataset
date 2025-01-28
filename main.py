import re
import nltk
from nltk.corpus import stopwords
import spacy
import pandas as pd
from textblob import TextBlob
import os
from html import unescape
import matplotlib
import Visualization
import Data_Cleaning

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


def calculate_sentiment(text):
    print("Calculating Sentiment")
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


# This function runs all the necessary steps to download the dataset from Kaggle and extract the contents and save
# them etc.
# this function is called by the one_and_done endpoint and when it starts running it will send a response back to the
# user that the operation has started.
def one_and_done():
    print("Starting the operation...")
    # data = data_io.read_tsv("amazon_reviews_us_Books_v1_02.tsv", Test_Path)
#
    # data = text_cleaning_and_preprocessing(data)
#
    # data['polarity'], data['subjectivity'] = zip(*data['review_body'].apply(calculate_sentiment))
#
    # data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)

    Visualization.correlation_charts_in_chunks(
        file_path='C:/Projects/Perform-EDA-on-textual-data/Data/cleaned_data_test_html.csv',
        chunk_size=10000,
        max_chunks=500
    )

    Visualization.word_cloud_in_chunks(
        file_path='C:/Projects/Perform-EDA-on-textual-data/Data/cleaned_data_test_html.csv',
        text_column='review_body',
        chunk_size=10000,
        max_chunks=500
    )

    Visualization.RAKE_in_chunks(
        file_path='C:/Projects/Perform-EDA-on-textual-data/Data/cleaned_data_test_html.csv',
        text_column='review_body',
        chunk_size=10000,
        max_chunks=500
    )

    Visualization.LDA_topic_modeling_in_chunks(
        file_path='C:/Projects/Perform-EDA-on-textual-data/Data/cleaned_data_test_html.csv',
        text_column='review_body',
        chunk_size=10000,
        max_chunks=500
    )

    Visualization.summarize_insights_and_visualize_in_chunks(
        file_path='C:/Projects/Perform-EDA-on-textual-data/Data/cleaned_data_test_html.csv',
        text_column='review_body',
        sentiment_column='star_rating',
        num_keywords=10,
        chunk_size=10000,
        max_chunks=10
    )

    Visualization.bigrams_by_sentiment_in_chunks(
        file_path='C:/Projects/Perform-EDA-on-textual-data/Data/cleaned_data_test_html.csv',
        text_column='review_body',
        sentiment_column='star_rating',
        chunk_size=10000,
        max_chunks=500
    )

    Visualization.trigrams_by_sentiment_in_chunks(
        file_path='C:/Projects/Perform-EDA-on-textual-data/Data/cleaned_data_test_html.csv',
        text_column='review_body',
        sentiment_column='star_rating',
        chunk_size=10000,
        max_chunks=500
    )

    print("Done")

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
# TODO: Write a function about reading the files
def text_cleaning_in_chunks(file_path, text_column):
    print("Text Cleaning in Chunks")
    cleaned_data = pd.DataFrame()
    pd_name_list = []
    pd_start_name = 'cleaned_data_'
    pd_end_name = '.csv'
    pd_num = 0
    # try:
    column_names = [text_column]
    for _, text_data in data_io.read_large_csv_with_dask(file_path, chunk_size=10000, columns=column_names):
        if text_data is not None:
            cleaned_data = text_cleaning_and_preprocessing(text_data)
            if cleaned_data is not None:
                pd_num += 1
                pd_name = pd_start_name + str(pd_num) + pd_end_name
                cleaned_data.to_csv(cleaned_file_path + pd_name, index=False)
                pd_name_list.append(pd_name)
        else:
            yield None

    # Connect all the cleaned data files and delete the individual files
    for pd_name in pd_name_list:
        if pd_name_list.index(pd_name) == 0:
            cleaned_data = pd.read_csv(cleaned_file_path + pd_name)
        else:
            cleaned_data = pd.concat([cleaned_data, pd.read_csv(cleaned_file_path + pd_name)], ignore_index=True)
        os.remove(cleaned_file_path + pd_name)
        # Save the final cleaned data
        cleaned_data.to_csv(cleaned_file_path + 'cleaned_data.csv', index=False)
    # except Exception as e:
    #     print(f"An error occurred while reading the CSV file in chunks: {e}")
    #     return None


if __name__ == '__main__':
    data_metadata_file_name = "Data_Info.txt"
    data_metadata_file_path = Test_Path + data_metadata_file_name

    # Read the data metadata and get the file names and tect column names which are seperated by a comma
    with open(data_metadata_file_path, 'r') as file:
        data_meta_data = file.read().replace('\n', ',')
    print(data_meta_data)
    data_meta_data = data_meta_data.split(",")
    # the file name and column names will be written in the file in the following format:
    # file_name1,column_name1,file_name2,column_name2,file_name3,column_name3
    file_name = []
    text_column = []
    for i in range(0, len(data_meta_data)-1, 2):
        file_name.append(data_meta_data[i])
        text_column.append(data_meta_data[i + 1])
    file_read_counter = 0
    # file_name length is the number of files to read
    max_files_to_read = len(file_name)

    Data_Cleaning.test()

    while file_read_counter < max_files_to_read:
        print("_" * 100)
        print("Reading file: ", file_name[file_read_counter])
        file_path = Test_Path + file_name[file_read_counter]
        data = text_cleaning_in_chunks(file_path, text_column=text_column[file_read_counter])
        file_read_counter += 1

    print("Done")
