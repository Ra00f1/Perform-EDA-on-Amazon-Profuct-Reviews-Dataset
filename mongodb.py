from pymongo import MongoClient
from pymongo.server_api import ServerApi
import datetime
import urllib.parse
import base64
import os

database_name = "amazon_reviews"
# collection name is date and time of the code execution
# Read the password from the file named password.txt
with open("password.txt", "r") as f:
    password = f.read()
uri = f"mongodb+srv://raoofagh:{password}@cluster0.sadjhyu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

def create_mongoDB_collection(collection_name):
    client = MongoClient('localhost', 27017)
    # create database first
    db = client[database_name]
    # create collection
    collection = db[collection_name]
    print("Collection created successfully!")
    return collection

def save_to_mongoDB(filename, collection_name):
    # Get all the images in the Output/ directory
    images = os.listdir("Output/" + filename)
    data = []
    for image in images:
        with open("Output/" + filename + "/" + image, "rb") as f:
            # Encode the image in base64
            encoded_image = base64.b64encode(f.read())
            data.append({
                "image_name": image,
                "image": encoded_image
            })
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[database_name]
    collection = db[collection_name]
    # Insert the data into the collection
    collection.insert_many(data)
    print("Data saved to MongoDB successfully!")
    return


def read_from_mongoDB(filename):
    os.makedirs("Output/" + filename + "Decoded")
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[database_name]
    collection = db[filename]
    data = collection.find()
    for d in data:
        # Decode the image from base64
        with open("Output/" + filename + "Decoded" + "/" + d["image_name"], "wb") as f:
            f.write(base64.b64decode(d["image"]))
    return

def get_all_collections():
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[database_name]
    collections = db.list_collection_names()
    return collections
