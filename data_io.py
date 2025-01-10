import os
import zipfile
import pandas as pd
import kagglehub
from main import text_cleaning_and_preprocessing
import dask.dataframe as dd

Test_Path = "Data/"

# download Amazon Customer Reviews Dataset from Kaggle
def download_dataset():
    # Download latest version
    path = kagglehub.dataset_download("cynthiarempel/amazon-us-customer-reviews-dataset")

    print("Path to dataset files:", path)


def unzip_data():
    file = "D:/Projects/Amazon Review/archive.zip"
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(Test_Path)


def read_tsv(file_name, file_path=Test_Path, max_rows=None):
    print("Reading file ", file_name)
    file_path = file_path + file_name
    # only read the first 1000 rows
    df = pd.read_csv(file_path, sep="\t", on_bad_lines='warn', low_memory=False, nrows=max_rows)
    print("File read successfully")
    return df


def read_and_connect_chunks(file_path=Test_Path):
    """
    Read all the chunks and connect them together to form a single DataFrame for further processing.
    :param file_path: path to the chunks
    :return: DataFrame
    """
    chunks = []
    file_path = file_path + "Chunks/"
    print(file_path)
    for file in os.listdir(file_path):
        print("Reading file: ", file)
        chunk = pd.read_csv(file_path + file, sep=",", on_bad_lines='warn', low_memory=False)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


# Read the dataset with default max_rows being all rows
def read_csv(file_path, columns=None):
    """
    Read the dataset with default max_rows being all rows if not specified.

    :param file_name: file name to read
    :param file_path: path to the file
    :param max_rows: maximum number of rows to read
    :return: DataFrame
    """

    print("Reading file ", file_path)
    # only read columns that are needed if specified
    if columns:
        ddf = dd.read_csv(file_path, sep=",", on_bad_lines='warn', low_memory=False, usecols=columns, dtype={'product_id': 'object'})
    else:
        ddf = dd.read_csv(file_path, sep=",", on_bad_lines='warn', low_memory=False, dtype={'product_id': 'object'})
    df = ddf.compute()
    print("File read successfully")
    return df


def get_dataset_shape(file_name, file_path=Test_Path, chunk_size=10000):
    """
    Get the shape of the dataset by reading it in chunks.
    :param file_name: file name to read
    :param file_path: path to the file
    :param chunk_size: size of each chunk
    :return: tuple
    """
    file_path = file_path + file_name
    total_rows = 0
    columns = None

    try:
        for chunk in pd.read_csv(file_path, sep=",", on_bad_lines='warn', low_memory=False, chunksize=chunk_size):
            if columns is None:
                columns = chunk.columns  # Capture column names from the first chunk
            total_rows += chunk.shape[0]  # Accumulate the number of rows

    except pd.errors.EmptyDataError:
        print("Reached the end of the file or encountered an empty chunk.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Return the calculated shape
    return (total_rows, len(columns) if columns is not None else 0)


def read_large_csv_with_dask(file_path, chunk_size=10000, columns=None, max_chunks=None):
    """
    Reads a large CSV file using Dask and processes a specific column if provided.

    Parameters:
    - :param file_path: str - Path to the CSV file.
    - :param chunk_size: int - Approximate number of rows per chunk.
    - :param columns: str or list - Name(s) of the text column(s) to process (optional).
    - :param max_chunks: int - Maximum number of chunks to process (optional).

    Yields:
    - pd.DataFrame: A chunk of data as a Pandas DataFrame.
    - list: A list of processed text from the specified column (if columns is provided).
    """
    global text_list
    try:
        # Ensure columns is a list
        if isinstance(columns, str):
            usecols = [columns]
        elif isinstance(columns, list):
            usecols = columns
        else:
            usecols = None

        # Load the CSV into a Dask DataFrame
        ddf = dd.read_csv(
            file_path,
            blocksize=chunk_size,
            assume_missing=True,
            sep=",",
            low_memory=False,
            usecols=usecols,
            dtype={'product_id': 'object'}  # Adjust dtype mapping as needed
        )

        # Number of chunks to process
        total_chunks = max_chunks if max_chunks else ddf.npartitions

        for i, partition in enumerate(ddf.to_delayed()[:total_chunks]):
            chunk = partition.compute()  # Convert the partition to a Pandas DataFrame

            if columns:
                # Ensure the specified column is string and handle NaNs
                if isinstance(columns, str):
                    chunk[columns] = chunk[columns].fillna("").astype(str)
                    text_list = chunk[columns].tolist()  # Convert the column to a list
                elif isinstance(columns, list):
                    for col in columns:
                        chunk[col] = chunk[col].fillna("").astype(str)
                    text_list = {col: chunk[col].tolist() for col in columns}
                yield chunk, text_list
            else:
                yield chunk, None

    except Exception as e:
        print(f"An error occurred while reading the CSV file with Dask: {e}")
