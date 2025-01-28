# Exploratory Data Analysis Project on Amazon US Customer Reviews Dataset
This project is an exploratory data analysis (EDA) initiative that will eventually transform into a large language model (LLM) project. The dataset used is the Amazon US Customer Reviews Dataset, which includes a diverse range of review types. These reviews often include mistakes, mispronunciations, and even entries that hold little value for extracting helpful data about a book or writer.

This project is a stepping stone to learning how LLMs work. However, before diving into LLM implementation, the goal is to gain a solid understanding of text data analysis and derive meaningful insights.

## Key Features
Dataset: The project leverages the Amazon US Customer Reviews Dataset, which provides extensive text data with varying levels of quality.

EDA Goals: To experiment with a variety of methods and visualizations to uncover helpful insights from the data.

Scalability: The project is designed to work efficiently with large datasets by using chunk-based processing with Dask, ensuring memory-efficient operations on limited hardware.
## Challenges
### 1. Identifying Effective Analysis Methods
The first challenge was determining which graphs and analytical methods would yield meaningful insights from the dataset. To address this:

Various methods and visualizations were tested, both self-developed and popular approaches found through research.
Future plans include adding even more methods to further expand the analysis toolkit.
### 2. Handling Large Dataset Size
The dataset's size posed a significant challenge for analysis on a personal computer. To mitigate this:

Dask was used for efficient reading and manipulation of large datasets.
Core functions were designed to process chunks of data using a yield mechanism, ensuring only manageable portions of data are processed at a time.
## Project Goals
Learn Text Analysis: Use the dataset to master text preprocessing and analysis techniques.

Extract Insights: Identify trends, patterns, and anomalies in customer reviews.

LLM Integration: Lay the foundation for transitioning the project to incorporate LLMs for advanced text understanding.

UI Connection: An API has been added to prepare for integration with a simple Android application that will serve as the project’s user interface in the future.
## Outputs and Insights
### Sample Outputs
Below are some of the insights generated during this project.
![wordcloud](https://github.com/user-attachments/assets/dabaf6f9-201e-4ea9-9b8d-d2d17449e5f5)
#### Figure 1: Word Cloud showing the most frequently used words in the reviews.

![Trigram](https://github.com/user-attachments/assets/954d0f3e-a130-4ffa-bda3-1a0f036ededf)
#### Figure 2: Bar chart of Trigrams for both negative and positive reviews.

## Next Steps
Experiment with LLMs to enhance feature extraction and text analysis.
Expand the API to provide additional functionalities for integration with the Android app.
Test and implement new EDA techniques to uncover deeper insights.
