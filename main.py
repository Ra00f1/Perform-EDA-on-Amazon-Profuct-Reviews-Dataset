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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import data_io
import matplotlib

matplotlib.use('TkAgg')  # Or use 'Agg' for non-interactive environments

# pandas disable low memory reading
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

Test_Path = "Data/"
Data_Path = "D:/Projects/Amazon Review/"

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')  # Load the Spacy English model


def explor_data():
    print("Reading files")
    for file in os.listdir(Data_Path):
        print("_" * 100)
        print(file)
        file_path = Data_Path + file
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


def calculate_sentiment(text):
    print("Calculating Sentiment")
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def categorize_sentiment(polarity):
    """
    Categorize the sentiment polarity score into positive, negative, or neutral.

    :param polarity: float - The sentiment polarity score.
    :return: str - The sentiment category (Positive, Negative, or Neutral).
    """
    print("Categorizing Sentiment")
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'


def word_cloud_from_large_data(file_path, text_column='review_body', min_frequency=0, chunk_size=100000):
    """
    Generate a word cloud from a large dataset by reading the data in chunks.
    A word cloud is a visualization technique that displays the most frequent words in a text corpus. The size of each
    word in the word cloud is proportional to its frequency in the text data.
    :param file_path: path to the file containing the text data.
    :param text_column: name of the column containing the text data.
    :param min_frequency: the minimum frequency threshold for filtering out words.
    :param chunk_size: the number of rows to read per chunk.
    :return: None
    """
    print("Generating Word Cloud for Large Data")

    word_freq = Counter()
    total_words = 0

    for _, text_data in data_io.read_large_csv_with_dask(file_path, chunk_size=chunk_size, text_column=text_column):
        # Tokenize the text data from the current chunk
        words = ' '.join(text_data).split()

        # Update the word frequency counter and total word count
        word_freq.update(words)
        total_words += len(words)

    # Filter out the words with low frequency
    filtered_words = {word: freq / total_words for word, freq in word_freq.items() if freq / total_words > min_frequency}

    # Generate the word cloud from the filtered words
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_font_size=80,
        colormap='viridis',
        contour_color='black',
        contour_width=1,
        prefer_horizontal=0.7
    ).generate_from_frequencies(filtered_words)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def word_cloud(data, min_frequency=0):
    """
    Generate a word cloud from the text data.
    A word cloud is a visualization technique that displays the most frequent words in a text corpus. The size of each
    word in the word cloud is proportional to its frequency in the text data.
    :param data: DataFrame containing the text data.(cleaned data)
    :param min_frequency: Minimum frequency threshold for filtering out words.
    :return: None
    """
    print("Generating Word Cloud")

    # Ensure all entries in 'review_body' are strings and drop NaN values
    data['review_body'] = data['review_body'].fillna("").astype(str)

    # Tokenize the text data
    words = ' '.join(data['review_body']).split()

    # Count the frequency of each word
    word_freq = Counter(words)
    # print(word_freq)

    # Filter out the words with low frequency
    total_words = sum(word_freq.values())
    filtered_words = {word: freq / total_words for word, freq in word_freq.items() if
                      freq / total_words > min_frequency}

    # Generate word cloud from the filtered words
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_font_size=80,
        colormap='viridis',
        contour_color='black',
        contour_width=1,
        prefer_horizontal=0.7
    ).generate_from_frequencies(filtered_words)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def correlation_charts(file_path, text_column=None, max_chunks=None):
    """
    Generate correlation charts to visualize the relationship between sentiment, ratings, and other numerical columns.
    This function reads the data in chunks to handle large datasets and generate the charts accordingly.

    :param file_path:
    :param text_column:
    :param max_chunks:
    :return:
    """
    print("Generating Correlation Charts")

    correlation_data = []
    for chunk, _ in data_io.read_large_csv_in_chunks(file_path, text_column=text_column, max_chunks=max_chunks):
        # Ensure required columns exist in the chunk
        required_columns = ['polarity', 'star_rating', 'helpful_votes', 'total_votes', 'subjectivity']

        # Some of the columns contain missing values, so we need to fill them with NaN
        for col in required_columns:
            if col not in chunk.columns:
                chunk[col] = np.nan  # Add missing columns with NaN

        # Prepare necessary columns for correlation charts
        chunk['sentiment_category'] = chunk['polarity'].apply(categorize_sentiment)
        correlation_data.append(chunk)

    # Concatenate all chunks into a single DataFrame for visualization
    data = pd.concat(correlation_data, ignore_index=True)

    # Scatterplot for sentiment polarity vs. ratings
    sns.scatterplot(x=data['polarity'], y=data['star_rating'])
    plt.title("Sentiment Polarity vs. Ratings")
    plt.xlabel("Polarity")
    plt.ylabel("Ratings")
    plt.show()

    # Correlation heatmap
    correlation_matrix = data[['polarity', 'star_rating']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation between Sentiment and Ratings")
    plt.show()

    # Bar chart of sentiment categories
    data['sentiment_category'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title("Distribution of Sentiment Categories")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Boxplot for sentiment polarity vs. ratings
    sns.boxplot(x=data['sentiment_category'], y=data['star_rating'], palette="viridis")
    plt.title("Ratings Distribution by Sentiment Category")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Star Rating")
    plt.show()

    # Correlation heatmap for numerical columns
    correlation_columns = ['star_rating', 'helpful_votes', 'total_votes', 'polarity', 'subjectivity']
    correlation_matrix = data[correlation_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap: Ratings, Votes, and Sentiment")
    plt.show()


def RAKE(file_path, text_column='review_body', max_chunks=None):
    """
    Perform keyword extraction using RAKE (Rapid Automatic Keyword Extraction) algorithm.
    RAKE is a keyword extraction algorithm that automatically extracts keywords from text by analyzing the frequency of
    word appearances and co-occurrences. It is particularly useful for identifying important keywords in text data.

    :param file_path:
    :param text_column:
    :param max_chunks:
    :return:
    """
    high_rated_reviews = []
    all_reviews = []

    for chunk, texts in data_io.read_large_csv_with_dask(file_path, text_column=text_column, max_chunks=max_chunks):
        # Filter high-rated reviews
        chunk_high_rated = chunk[chunk['star_rating'] >= 4][text_column].dropna().str.strip()
        chunk_all_reviews = chunk[text_column].dropna().str.strip()

        # Only add non-empty and non-stopword-only reviews
        high_rated_reviews.extend(chunk_high_rated[chunk_high_rated != ""].tolist())
        all_reviews.extend(chunk_all_reviews[chunk_all_reviews != ""].tolist())

    # Ensure there are valid reviews
    if not high_rated_reviews:
        print("No valid high-rated reviews found.")
        return

    if not all_reviews:
        print("No valid reviews found.")
        return

    # TF-IDF Vectorizer for high-rated reviews
    try:
        tfidf = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(high_rated_reviews)

        # Display TF-IDF results
        tfidf_scores = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=tfidf.get_feature_names_out()
        ).sum().sort_values(ascending=False)
        print("Top Keywords in High-Rated Reviews:")
        print(tfidf_scores.head(10))

        # Visualize TF-IDF scores
        tfidf_scores.head(10).plot(kind='bar', figsize=(10, 5), color='skyblue')
        plt.title("Top Keywords in High-Rated Reviews (TF-IDF)")
        plt.xlabel("Keywords")
        plt.ylabel("TF-IDF Score")
        plt.xticks(rotation=45)
        plt.show()

    except ValueError as e:
        print(f"Error in TF-IDF Vectorizer: {e}")
        return

    # RAKE analysis for high-rated reviews
    rake = Rake()
    rake.extract_keywords_from_sentences(high_rated_reviews)
    rake_keywords = rake.get_ranked_phrases_with_scores()

    print("Top Keywords from High-Rated Reviews (RAKE):")
    print(rake_keywords[:10])

    rake_df = pd.DataFrame(rake_keywords[:10], columns=['Score', 'Keyword'])
    rake_df.plot(x='Keyword', y='Score', kind='barh', figsize=(10, 6), color='orange', legend=False)
    plt.title("Top Keywords in High-Rated Reviews (RAKE)")
    plt.xlabel("RAKE Score")
    plt.ylabel("Keywords")
    plt.gca().invert_yaxis()
    plt.show()

    # CountVectorizer for topic modeling
    try:
        count_vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        count_matrix = count_vectorizer.fit_transform(all_reviews)

        lda = LatentDirichletAllocation(n_components=5, random_state=42, learning_method='batch')
        lda.fit(count_matrix)

        def display_topics(model, feature_names, num_words):
            for idx, topic in enumerate(model.components_):
                print(f"Topic {idx + 1}:")
                print([feature_names[i] for i in topic.argsort()[-num_words:][::-1]])
                print()

        display_topics(lda, count_vectorizer.get_feature_names_out(), 10)

        doc_topic_matrix = lda.transform(count_matrix)
        dominant_topics = np.argmax(doc_topic_matrix, axis=1)
        topic_counts = pd.Series(dominant_topics).value_counts()

        topic_counts.plot(kind='bar', figsize=(8, 5), color='purple')
        plt.title("Number of Reviews per Topic")
        plt.xlabel("Topic")
        plt.ylabel("Number of Reviews")
        plt.xticks(rotation=0)
        plt.show()

    except ValueError as e:
        print(f"Error in LDA Topic Modeling: {e}")


def LDA_topic_modeling(reviews, num_topics=5, num_keywords=10):
    """
    Perform topic modeling on reviews using LDA.
    What is LDA?
    Latent Dirichlet Allocation (LDA) is a generative probabilistic model for topic modeling. It assumes that each
    document is a mixture of topics and that each word in the document is attributable to one of the document's topics.
    LDA can be used to identify the main topics in a collection of text data.

    Parameters:
    - reviews: list of str - The text data for topic modeling.
    - num_topics: int - Number of topics to extract.
    - num_keywords: int - Number of keywords to display for each topic.

    Returns:
    - None
    """
    print("-" * 100)
    print("Performing Topic Modeling using LDA")
    # Step 1: Clean the input data
    print("Cleaning the input data")
    reviews = [review if isinstance(review, str) else "" for review in reviews]  # Replace non-strings with empty strings
    reviews = [review for review in reviews if review.strip()]  # Remove empty or whitespace-only strings

    # Step 2: Convert text to document-term matrix
    print("Converting text to document-term matrix")
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    dtm = vectorizer.fit_transform(reviews)

    # Step 3: Apply LDA
    print("Applying Latent Dirichlet Allocation( )")
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    # Step 4: Extract and display topics
    print("Extracting and displaying topics")
    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    print("Topics Identified:\n")
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords = [feature_names[i] for i in topic.argsort()[-num_keywords:][::-1]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_keywords)}")
        topics[f"Topic {topic_idx + 1}"] = top_keywords

    # Step 5: Visualize Topic Distributions
    doc_topic_matrix = lda.transform(dtm)
    dominant_topics = pd.DataFrame(doc_topic_matrix).idxmax(axis=1).value_counts()

    plt.figure(figsize=(10, 6))
    dominant_topics.sort_index().plot(kind='bar', color='skyblue')
    plt.title("Number of Reviews per Topic")
    plt.xlabel("Topic")
    plt.ylabel("Number of Reviews")
    plt.xticks(range(len(topics)), labels=topics.keys(), rotation=45)
    plt.tight_layout()
    plt.show()


def summarize_insights_and_visualize(
    data, text_column='review_body', sentiment_column='star_rating', num_keywords=10
):
    """
    Summarize actionable insights by extracting and visualizing top keywords for positive and negative sentiments.

    Parameters:
    - data: pd.DataFrame - The dataset containing reviews and sentiment ratings.
    - text_column: str - The column containing review text.
    - sentiment_column: str - The column containing sentiment ratings.
    - num_keywords: int - Number of top keywords to display.

    Returns:
    - None
    """
    # Step 1: Filter and clean data
    data[text_column] = data[text_column].fillna("").astype(str)  # Replace NaN with empty strings

    # Split reviews into positive and negative sentiments
    positive_reviews = data[data[sentiment_column] >= 4][text_column].tolist()
    negative_reviews = data[data[sentiment_column] <= 2][text_column].tolist()

    # Step 2: Define TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))

    # Step 3: Extract keywords for positive reviews
    positive_matrix = tfidf.fit_transform(positive_reviews)
    positive_keywords = pd.DataFrame(positive_matrix.toarray(), columns=tfidf.get_feature_names_out()).sum().sort_values(ascending=False)

    # Step 4: Extract keywords for negative reviews
    negative_matrix = tfidf.fit_transform(negative_reviews)
    negative_keywords = pd.DataFrame(negative_matrix.toarray(), columns=tfidf.get_feature_names_out()).sum().sort_values(ascending=False)

    # Step 5: Reduce overlap between positive and negative keywords
    positive_set = set(positive_keywords.head(2 * num_keywords).index)
    negative_set = set(negative_keywords.head(2 * num_keywords).index)
    exclusive_positive = positive_set - negative_set
    exclusive_negative = negative_set - positive_set

    # Filter exclusive keywords
    positive_keywords = positive_keywords[positive_keywords.index.isin(exclusive_positive)].head(num_keywords)
    negative_keywords = negative_keywords[negative_keywords.index.isin(exclusive_negative)].head(num_keywords)

    # Step 6: Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Positive sentiment keywords
    positive_keywords.plot(kind='bar', ax=axes[0], color='green')
    axes[0].set_title("Top Keywords in Positive Sentiment Reviews")
    axes[0].set_xlabel("Keywords")
    axes[0].set_ylabel("TF-IDF Score")
    axes[0].tick_params(axis='x', rotation=45)

    # Negative sentiment keywords
    negative_keywords.plot(kind='bar', ax=axes[1], color='red')
    axes[1].set_title("Top Keywords in Negative Sentiment Reviews")
    axes[1].set_xlabel("Keywords")
    axes[1].set_ylabel("TF-IDF Score")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Step 7: Summarize actionable insights
    print("Actionable Insights:")
    print("\nPositive Sentiment Insights:")
    print(positive_keywords)

    print("\nNegative Sentiment Insights:")
    print(negative_keywords)


# This function runs all the necessary steps to download the dataset from Kaggle and extract the contents and save
# them etc.
def one_and_done():
    data = data_io.read_tsv("amazon_reviews_us_Books_v1_02.tsv", Test_Path)

    data = text_cleaning_and_preprocessing(data)

    data['polarity'], data['subjectivity'] = zip(*data['review_body'].apply(calculate_sentiment))

    data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)

    correlation_charts(Test_Path + "cleaned_data_test_html.csv")

    word_cloud(data)

    RAKE(Test_Path + "cleaned_data_test_html.csv")

    LDA_topic_modeling(data['review_body'].tolist())

    summarize_insights_and_visualize(data, num_keywords=50)

    print("Done")


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
    #                          chunk_num=35)
    # data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)
    # # Analyze the sentiment of reviews and add polarity and subjectivity scores to the DataFrame
#
    # data['polarity'], data['subjectivity'] = zip(*data['review_body'].apply(calculate_sentiment))
#
    # # Save the cleaned data
    # data.to_csv(Test_Path + "cleaned_data_test_html.csv", index=False)
#
    # # Analyze sentiment of reviews and add the scores to the DataFrame
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

    # Load cleaned data
    # data = data_io.read_csv("cleaned_data_test_html.csv", Test_Path)
    # data_info(data)

    # data = data_io.read_and_connect_chunks(Test_Path)
#
    # try:
    #     word_cloud(data)
    # except:
    #     print("Word cloud error")
    #     print(data['review_body'])
#
    # data['polarity'], data['subjectivity'] = zip(*data['review_body'].apply(calculate_sentiment))
    # print("Sentiment scores added (Polarity and Subjectivity)")
#
    # data.to_csv(Test_Path + "cleaned_data4.csv", index=False)
#
    # correlation_charts(data)
#
    # RAKE(data)

    file_name = "cleaned_data_test_html.csv"
    # word_cloud_from_large_data(file_path)

    # shape = data_io.get_dataset_shape("cleaned_data_test_html.csv", Test_Path)

    # print(shape)

    # correlation_charts(file_path)
    # TODO: creates topics of entire sentences and not just words
    # RAKE(file_path, max_chunks=10)

    data = data_io.read_csv(file_name, Test_Path)

    # LDA_topic_modeling(data['review_body'].tolist())

    summarize_insights_and_visualize(data, num_keywords=50)

    print("Done")
