from fastapi import FastAPI
from pydantic import BaseModel
import threading
import main
import mongodb
import datetime

app = FastAPI()

class Message(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/start_operation")
def start_operation():
    # Inform the user that the operation has started
    response = Message(message="The operation has started...")

    # Create a folder with the name being the time and date of the run
    filename = datetime.datetime.now().strftime("Y%YM%mD%d_H%HM%MS%S")

    # Run the operations in the background (without using threading)
    main.one_and_done(filename)

    # after the thread above is finished, save the data to MongoDB
    mongodb.save_to_mongoDB(filename, filename)

    return response

@app.get("/get_data")
def get_data():
    # Get all the collections in the database
    collections = mongodb.get_all_collections()
    # convert collections to string
    collections = str(collections)
    response = Message(message="All the collections are: " + collections + " TO load the data type load_data/{"
                                                                         "collection_name}")
    return response


@app.get("/load_data/{collection_name}")
def load_data(collection_name: str):
    # Load the data from the specified collection
    data = mongodb.read_from_mongoDB(collection_name)
    response = Message(message="Data loaded successfully from collection: " + collection_name)
    return response
