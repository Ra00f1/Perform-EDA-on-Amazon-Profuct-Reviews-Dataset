import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from rake_nltk import Rake
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import data_io

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

    Large data version of the word cloud function that reads the data in chunks.

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

    Normal version of the word cloud function that takes a DataFrame as input.

    :param data: DataFrame containing the text data.(cleaned data)(Columns needed: review_body, star_rating)
    :param min_frequency: Minimum frequency threshold for filtering out words.
    :return: None
    """
    print("Generating Word Cloud")

    # Ensure all entries in 'review_body' are strings and drop NaN values
    data['review_body'] = data['review_body'].fillna("").astype(str)

    positive_reviews = data[data['star_rating'] >= 4]['review_body'].tolist()
    negative_reviews = data[data['star_rating'] <= 2]['review_body'].tolist()

    # Tokenize the text data
    positive_reviews = ' '.join(data['review_body']).split()
    negative_reviews = ' '.join(data['review_body']).split()

    # Count the frequency of each word
    word_freq_pos = Counter(positive_reviews)
    word_freq_neg = Counter(negative_reviews)
    # print(word_freq)

    # Filter out the words with low frequency
    total_words = sum(word_freq_pos.values())
    filtered_positive_reviews = {word: freq / total_words for word, freq in word_freq_pos.items() if
                      freq / total_words > min_frequency}
    total_words = sum(word_freq_neg.values())
    filtered_negative_reviews = {word: freq / total_words for word, freq in word_freq_neg.items() if
                      freq / total_words > min_frequency}

    # Generate word cloud from the filtered words
    wordcloud_positive = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_font_size=80,
        colormap='viridis',
        contour_color='black',
        contour_width=1,
        prefer_horizontal=0.7
    ).generate_from_frequencies(filtered_positive_reviews)
    wordcloud_negative = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_font_size=80,
        colormap='viridis',
        contour_color='black',
        contour_width=1,
        prefer_horizontal=0.7
    ).generate_from_frequencies(filtered_negative_reviews)

    # Display the word cloud
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(wordcloud_positive, interpolation="bilinear")
    axes[0].set_title("Word Cloud for Positive Reviews")
    axes[0].axis('off')

    axes[1].imshow(wordcloud_negative, interpolation="bilinear")
    axes[1].set_title("Word Cloud for Negative Reviews")
    axes[1].axis('off')

    plt.show()


def word_cloud_quick(data):
    """
    Generate a word cloud from the text data.
    A word cloud is a visualization technique that displays the most frequent words in a text corpus. The size of each
    word in the word cloud is proportional to its frequency in the text data.

    Quick version of the word cloud function that takes a list of strings as input.

    :param data: List of strings containing the text data.
    :return: None
    """
    print("Generating Word Cloud")

    # Generate word cloud from the text data
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_font_size=80,
        colormap='viridis',
        contour_color='black',
        contour_width=1,
        prefer_horizontal=0.7
    ).generate(' '.join(data))

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def correlation_charts(file_path, text_column=None, max_chunks=None):
    """
    Generate correlation charts to visualize the relationship between sentiment, ratings, and other numerical columns.
    This function reads the data in chunks to handle large datasets and generate the charts accordingly.

    :param file_path: str - Path to the file containing the data.
    :param text_column: str - Name of the text column containing the reviews.
    :param max_chunks: int - Maximum number of chunks to process.
    :return: None
    """
    print("Generating Correlation Charts")

    correlation_data = []
    for chunk, _ in data_io.read_large_csv_with_dask(file_path, text_column=text_column, max_chunks=max_chunks):
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

    :param file_path: str - Path to the file containing the data.
    :param text_column: str - Name of the text column containing the reviews.
    :param max_chunks: int - Maximum number of chunks to process.
    :return: None
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
    This function uses TF-IDF to extract keywords from reviews and visualize the top keywords for positive and negative
    sentiment ratings. It also displays word clouds for positive and negative keywords to provide a visual representation
    of the insights.

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
    # How does it do that?
    positive_set = set(positive_keywords.head(2 * num_keywords).index)
    negative_set = set(negative_keywords.head(2 * num_keywords).index)

    # Removing the keywords that are common between positive and negative reviews.
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

    # Step 7: Show word cloud for positive and negative keywords
    word_cloud_quick(positive_keywords.index)
    word_cloud_quick(negative_keywords.index)

    # Step 8: Summarize actionable insights
    print("Actionable Insights:")
    print("\nPositive Sentiment Insights:")
    print(positive_keywords)

    print("\nNegative Sentiment Insights:")
    print(negative_keywords)