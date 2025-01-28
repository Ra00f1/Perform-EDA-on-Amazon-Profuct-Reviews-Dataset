import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'


def word_cloud_in_chunks(file_path, text_column='review_body', min_frequency=0, chunk_size=100000, max_chunks=None):
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

    for _, text_data in data_io.read_large_csv_with_dask(file_path, chunk_size=chunk_size, columns=text_column,
                                                         max_chunks=max_chunks):
        # Tokenize the text data from the current chunk
        words = ' '.join(text_data).split()

        # Update the word frequency counter and total word count
        word_freq.update(words)
        total_words += len(words)

    # Filter out the words with low frequency
    filtered_words = {word: freq / total_words for word, freq in word_freq.items() if
                      freq / total_words > min_frequency}

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

    plt.savefig('Output/wordcloud.png')
    plt.close()


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


def correlation_charts_in_chunks(file_path, chunk_size=10000, max_chunks=None):
    """
    Generate correlation charts to visualize the relationship between sentiment, ratings, and other numerical columns.
    This function reads the data in chunks to handle large datasets and generates the charts accordingly.

    :param file_path: str - Path to the file containing the data.
    :param text_column: str - Name of the text column containing the reviews (optional).
    :param chunk_size: int - Number of rows to process per chunk.
    :param max_chunks: int - Maximum number of chunks to process.
    :return: None
    """
    print("Generating Correlation Charts with Chunked Processing")

    correlation_data = []

    # Read and process the file in chunks
    for chunk, _ in data_io.read_large_csv_with_dask(
            file_path=file_path,
            chunk_size=chunk_size,
            columns=None,  # Load all columns
            max_chunks=max_chunks
    ):
        # Ensure required columns exist in the chunk
        required_columns = ['polarity', 'star_rating', 'helpful_votes', 'total_votes', 'subjectivity']

        for col in required_columns:
            if col not in chunk.columns:
                chunk[col] = np.nan

        chunk['sentiment_category'] = chunk['polarity'].apply(categorize_sentiment)
        correlation_data.append(chunk)

    # Concatenate all chunks into a single DataFrame
    data = pd.concat(correlation_data, ignore_index=True)

    # Scatterplot for sentiment polarity vs. ratings
    sns.scatterplot(x=data['polarity'], y=data['star_rating'])
    plt.title("Sentiment Polarity vs. Ratings")
    plt.xlabel("Polarity")
    plt.ylabel("Ratings")

    plt.savefig('Output/Senti_vs_Rating.png')
    plt.close()

    # Correlation heatmap
    correlation_matrix = data[['polarity', 'star_rating']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation between Sentiment and Ratings")

    plt.savefig('Output/Senti_vs_Rating_heatmap.png')
    plt.close()

    # Bar chart of sentiment categories
    data['sentiment_category'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title("Distribution of Sentiment Categories")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    plt.savefig('Output/Senti_category.png')
    plt.close()

    # Boxplot for sentiment polarity vs. ratings
    sns.boxplot(x=data['sentiment_category'], y=data['star_rating'], palette="viridis")
    plt.title("Ratings Distribution by Sentiment Category")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Star Rating")

    plt.savefig('Output/Rating_by_Senti.png')
    plt.close()

    # Correlation heatmap for numerical columns
    correlation_columns = ['star_rating', 'helpful_votes', 'total_votes', 'polarity', 'subjectivity']
    correlation_matrix = data[correlation_columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap: Ratings, Votes, and Sentiment")

    plt.savefig('Output/Correlation_heatmap.png')
    plt.close()


def RAKE_in_chunks(file_path, text_column='review_body', sentiment_column='star_rating', chunk_size=10000,
                   max_chunks=None):
    """
    Perform keyword extraction using RAKE (Rapid Automatic Keyword Extraction) algorithm on chunked data.

    :param file_path: str - Path to the file containing the data.
    :param text_column: str - Name of the text column containing the reviews.
    :param sentiment_column: str - Name of the column containing sentiment ratings.
    :param chunk_size: int - Number of rows to process per chunk.
    :param max_chunks: int - Maximum number of chunks to process.
    :return: None
    """
    high_rated_reviews = []
    all_reviews = []

    # Read and process the file in chunks
    for chunk, _ in data_io.read_large_csv_with_dask(
            file_path=file_path,
            chunk_size=chunk_size,
            columns=[text_column, sentiment_column],
            max_chunks=max_chunks
    ):
        # Filter high-rated reviews
        chunk[sentiment_column] = pd.to_numeric(chunk[sentiment_column], errors='coerce')
        chunk_high_rated = chunk[chunk[sentiment_column] >= 4][text_column].dropna().str.strip()
        chunk_all_reviews = chunk[text_column].dropna().str.strip()

        high_rated_reviews.extend(chunk_high_rated[chunk_high_rated != ""].tolist())
        all_reviews.extend(chunk_all_reviews[chunk_all_reviews != ""].tolist())

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

        tfidf_scores.head(10).plot(kind='bar', figsize=(10, 5), color='skyblue')
        plt.title("Top Keywords in High-Rated Reviews (TF-IDF)")
        plt.xlabel("Keywords")
        plt.ylabel("TF-IDF Score")
        plt.xticks(rotation=45)


        plt.savefig('Output/TFIDF.png')
        plt.close()

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


    plt.savefig('Output/RAKE.png')
    plt.close()

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


        plt.savefig('Output/Topic.png')
        plt.close()

    except ValueError as e:
        print(f"Error in LDA Topic Modeling: {e}")


def LDA_topic_modeling_in_chunks(file_path, text_column='review_body', num_topics=5, num_keywords=10, chunk_size=10000,
                                 max_chunks=None):
    """
    Perform topic modeling on reviews using LDA with chunked processing for large datasets.

    Parameters:
    - file_path: str - Path to the CSV file containing the reviews.
    - text_column: str - Column name containing the reviews.
    - num_topics: int - Number of topics to extract.
    - num_keywords: int - Number of keywords to display for each topic.
    - chunk_size: int - Number of rows to process per chunk.
    - max_chunks: int - Maximum number of chunks to process (optional).

    Returns:
    - None
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import pandas as pd
    import matplotlib.pyplot as plt

    print("-" * 100)
    print("Performing Topic Modeling using LDA with Chunked Processing")

    # Step 1: Build a global vocabulary
    print("Building global vocabulary")
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    reviews_list = []

    # First pass to collect reviews for vocabulary building
    for chunk, _ in data_io.read_large_csv_with_dask(
            file_path=file_path,
            chunk_size=chunk_size,
            columns=text_column,
            max_chunks=max_chunks
    ):
        reviews = chunk[text_column].fillna("").astype(str).tolist()
        reviews = [review.strip() for review in reviews if review.strip()]  # Remove empty or whitespace-only strings
        reviews_list.extend(reviews)

    if not reviews_list:
        print("No valid reviews found.")
        return

    # Fit the vectorizer to the global reviews list
    vectorizer.fit(reviews_list)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # Step 2: Process data in chunks with the global vocabulary
    print("Processing data in chunks with the global vocabulary")
    dtm_list = []

    for chunk, _ in data_io.read_large_csv_with_dask(
            file_path=file_path,
            chunk_size=chunk_size,
            columns=text_column,
            max_chunks=max_chunks
    ):
        reviews = chunk[text_column].fillna("").astype(str).tolist()
        reviews = [review.strip() for review in reviews if review.strip()]  # Remove empty or whitespace-only strings

        if reviews:
            dtm_chunk = vectorizer.transform(reviews)  # Use the global vocabulary
            dtm_list.append(dtm_chunk)

    # Combine all chunked DTM matrices
    if not dtm_list:
        print("No valid reviews found after chunk processing.")
        return

    dtm = scipy.sparse.vstack(dtm_list)  # Combine matrices with consistent shapes
    print(f"Final DTM shape: {dtm.shape}")

    # Step 3: Apply LDA
    print("Applying Latent Dirichlet Allocation (LDA)")
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
    print("Visualizing Topic Distributions")
    doc_topic_matrix = lda.transform(dtm)
    dominant_topics = pd.DataFrame(doc_topic_matrix).idxmax(axis=1).value_counts()

    plt.figure(figsize=(10, 6))
    dominant_topics.sort_index().plot(kind='bar', color='skyblue')
    plt.title("Number of Reviews per Topic")
    plt.xlabel("Topic")
    plt.ylabel("Number of Reviews")
    plt.xticks(range(len(topics)), labels=topics.keys(), rotation=45)
    plt.tight_layout()


    plt.savefig('Output/Topic_distribution.png')
    plt.close()


def summarize_insights_and_visualize_in_chunks(
        file_path, text_column='review_body', sentiment_column='star_rating', num_keywords=10, chunk_size=10000,
        max_chunks=None
):
    """
    Summarize actionable insights by extracting and visualizing top keywords for positive and negative sentiments,
    processing the data in chunks using Dask.

    Parameters:
    - file_path: str - Path to the CSV file.
    - text_column: str - The column containing review text.
    - sentiment_column: str - The column containing sentiment ratings.
    - num_keywords: int - Number of top keywords to display.
    - chunk_size: int - Approximate number of rows per chunk.
    - max_chunks: int - Maximum number of chunks to process (optional).

    Returns:
    - None
    """
    from collections import Counter
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd

    # Initialize counters for TF-IDF keyword aggregation
    positive_counter = Counter()
    negative_counter = Counter()

    # Process the file in chunks
    for chunk, _ in data_io.read_large_csv_with_dask(
            file_path=file_path,
            chunk_size=chunk_size,
            columns=[text_column, sentiment_column],
            max_chunks=max_chunks
    ):
        chunk[sentiment_column] = pd.to_numeric(chunk[sentiment_column], errors='coerce')

        # Filter and clean the chunk
        positive_reviews = chunk[chunk[sentiment_column] >= 4][text_column].tolist()
        negative_reviews = chunk[chunk[sentiment_column] <= 2][text_column].tolist()

        # Define TF-IDF Vectorizer
        tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))

        # Extract keywords for positive reviews
        if positive_reviews:
            positive_matrix = tfidf.fit_transform(positive_reviews)
            positive_keywords = pd.DataFrame(
                positive_matrix.toarray(), columns=tfidf.get_feature_names_out()
            ).sum().sort_values(ascending=False)
            positive_counter.update(positive_keywords.to_dict())

        # Extract keywords for negative reviews
        if negative_reviews:
            negative_matrix = tfidf.fit_transform(negative_reviews)
            negative_keywords = pd.DataFrame(
                negative_matrix.toarray(), columns=tfidf.get_feature_names_out()
            ).sum().sort_values(ascending=False)
            negative_counter.update(negative_keywords.to_dict())

    # Convert counters to sorted keyword lists
    positive_keywords = pd.Series(positive_counter).sort_values(ascending=False)
    negative_keywords = pd.Series(negative_counter).sort_values(ascending=False)

    # Reduce overlap between positive and negative keywords
    positive_set = set(positive_keywords.head(2 * num_keywords).index)
    negative_set = set(negative_keywords.head(2 * num_keywords).index)
    exclusive_positive = positive_set - negative_set
    exclusive_negative = negative_set - positive_set

    # Filter exclusive keywords
    positive_keywords = positive_keywords[positive_keywords.index.isin(exclusive_positive)].head(num_keywords)
    negative_keywords = negative_keywords[negative_keywords.index.isin(exclusive_negative)].head(num_keywords)

    # Visualize the results
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


    plt.savefig('Output/Insights.png')
    plt.close()

    # Summarize actionable insights
    print("Actionable Insights:")
    print("\nPositive Sentiment Insights:")
    print(positive_keywords)

    print("\nNegative Sentiment Insights:")
    print(negative_keywords)


def bigrams_by_sentiment_in_chunks(
        file_path,
        text_column,
        sentiment_column,
        chunk_size=10000,
        max_chunks=None,
        num_bigrams=15,
):
    """
    Extract and plot bigram frequencies for positive and negative sentiments in chunks.

    Parameters:
    - file_path: str - Path to the CSV file.
    - text_column: str - Name of the text column to process.
    - sentiment_column: str - Name of the sentiment column to divide data.
    - chunk_size: int - Number of rows per chunk.
    - max_chunks: int - Maximum number of chunks to process (optional).
    - num_bigrams: int - Number of top bigrams to display.

    Returns:
    - None: Displays bar plots of bigram frequencies for positive and negative sentiments.
    """
    positive_bigram_counter = Counter()
    negative_bigram_counter = Counter()

    try:
        # Process the file in chunks
        for i, (chunk, text_data) in enumerate(
                data_io.read_large_csv_with_dask(file_path, chunk_size, [text_column, sentiment_column], max_chunks)
        ):
            print(f"Processing chunk {i + 1}...")

            # Ensure sentiment column is numeric for comparison
            chunk[sentiment_column] = pd.to_numeric(chunk[sentiment_column], errors="coerce")

            # Split data into positive and negative based on sentiment threshold
            positive_data = chunk.loc[chunk[sentiment_column] >= 4, text_column].tolist()
            negative_data = chunk.loc[chunk[sentiment_column] <= 2, text_column].tolist()

            # Create bigram counters for positive and negative reviews
            cv = CountVectorizer(ngram_range=(2, 2))

            if positive_data:
                X_positive = cv.fit_transform(positive_data)
                positive_bigram_freq = Counter(
                    {word: X_positive[:, idx].sum() for word, idx in cv.vocabulary_.items()}
                )
                positive_bigram_counter.update(positive_bigram_freq)

            if negative_data:
                X_negative = cv.fit_transform(negative_data)
                negative_bigram_freq = Counter(
                    {word: X_negative[:, idx].sum() for word, idx in cv.vocabulary_.items()}
                )
                negative_bigram_counter.update(negative_bigram_freq)

        # Prepare data for plotting
        positive_bigrams = positive_bigram_counter.most_common(num_bigrams)
        negative_bigrams = negative_bigram_counter.most_common(num_bigrams)

        positive_df = pd.DataFrame(positive_bigrams, columns=["Bigram", "Frequency"])
        negative_df = pd.DataFrame(negative_bigrams, columns=["Bigram", "Frequency"])

        positive_df = positive_df.sort_values(by="Frequency", ascending=True)
        negative_df = negative_df.sort_values(by="Frequency", ascending=True)

        # delete the similar bigrams from both positive and negative
        positive_exclusive_df = positive_df[~positive_df["Bigram"].isin(negative_df["Bigram"])]
        negative_exclusive_df = negative_df[~negative_df["Bigram"].isin(positive_df["Bigram"])]

        # Plot side-by-side bar graphs
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Positive bigrams
        positive_exclusive_df.plot(kind="barh", x="Bigram", y="Frequency", ax=axes[0], color="green")
        axes[0].set_title("Top Positive Bigrams")
        axes[0].set_xlabel("Frequency")
        axes[0].set_ylabel("Bigram")
        axes[0].tick_params(axis="y", labelrotation=0)

        # Negative bigrams
        negative_exclusive_df.plot(kind="barh", x="Bigram", y="Frequency", ax=axes[1], color="red")
        axes[1].set_title("Top Negative Bigrams")
        axes[1].set_xlabel("Frequency")
        axes[1].set_ylabel("Bigram")
        axes[1].tick_params(axis="y", labelrotation=0)

        plt.tight_layout()


        plt.savefig('Output/Bigram.png')
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")


def trigrams_by_sentiment_in_chunks(
        file_path,
        text_column,
        sentiment_column,
        chunk_size=10000,
        max_chunks=None,
        num_triograms=15,
):
    """
    Extract and plot triogram frequencies for positive and negative sentiments in chunks.

    Parameters:
    - file_path: str - Path to the CSV file.
    - text_column: str - Name of the text column to process.
    - sentiment_column: str - Name of the sentiment column to divide data.
    - chunk_size: int - Number of rows per chunk.
    - max_chunks: int - Maximum number of chunks to process (optional).
    - num_triograms: int - Number of top triograms to display.

    Returns:
    - None: Displays bar plots of triogram frequencies for positive and negative sentiments.
    """
    positive_trigram_counter = Counter()
    negative_trigram_counter = Counter()

    try:
        # Process the file in chunks
        for i, (chunk, text_data) in enumerate(
                data_io.read_large_csv_with_dask(file_path, chunk_size, [text_column, sentiment_column], max_chunks)
        ):
            print(f"Processing chunk {i + 1}...")

            # Ensure sentiment column is numeric for comparison
            chunk[sentiment_column] = pd.to_numeric(chunk[sentiment_column], errors="coerce")

            # Split data into positive and negative based on sentiment threshold
            positive_data = chunk.loc[chunk[sentiment_column] >= 4, text_column].tolist()
            negative_data = chunk.loc[chunk[sentiment_column] <= 2, text_column].tolist()

            # Create trigram counters for positive and negative reviews
            cv = CountVectorizer(ngram_range=(3, 3))

            if positive_data:
                X_positive = cv.fit_transform(positive_data)
                positive_trigram_freq = Counter(
                    {word: X_positive[:, idx].sum() for word, idx in cv.vocabulary_.items()}
                )
                positive_trigram_counter.update(positive_trigram_freq)

            if negative_data:
                X_negative = cv.fit_transform(negative_data)
                negative_trigram_freq = Counter(
                    {word: X_negative[:, idx].sum() for word, idx in cv.vocabulary_.items()}
                )
                negative_trigram_counter.update(negative_trigram_freq)

        # Prepare data for plotting
        positive_trigrams = positive_trigram_counter.most_common(num_triograms)
        negative_trigrams = negative_trigram_counter.most_common(num_triograms)

        positive_df = pd.DataFrame(positive_trigrams, columns=["Trigram", "Frequency"])
        negative_df = pd.DataFrame(negative_trigrams, columns=["Trigram", "Frequency"])

        positive_df = positive_df.sort_values(by="Frequency", ascending=True)
        negative_df = negative_df.sort_values(by="Frequency", ascending=True)

        # delete the similar trigrams from both positive and negative
        positive_exclusive_df = positive_df[~positive_df["Trigram"].isin(negative_df["Trigram"])]
        negative_exclusive_df = negative_df[~negative_df["Trigram"].isin(positive_df["Trigram"])]

        # Plot side-by-side bar graphs
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Positive trigrams
        positive_exclusive_df.plot(kind="barh", x="Trigram", y="Frequency", ax=axes[0], color="green")
        axes[0].set_title("Top Positive Trigrams")
        axes[0].set_xlabel("Frequency")
        axes[0].set_ylabel("Trigram")
        axes[0].tick_params(axis="y", labelrotation=0)

        # Negative trigrams
        negative_exclusive_df.plot(kind="barh", x="Trigram", y="Frequency", ax=axes[1], color="red")
        axes[1].set_title("Top Negative Trigrams")
        axes[1].set_xlabel("Frequency")
        axes[1].set_ylabel("Trigram")
        axes[1].tick_params(axis="y", labelrotation=0)

        plt.tight_layout()


        plt.savefig('Output/Trigram.png')
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")
