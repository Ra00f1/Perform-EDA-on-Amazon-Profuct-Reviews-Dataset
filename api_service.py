from fastapi import FastAPI
from pydantic import BaseModel
import threading
import main

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

    # Run the operations in the background (using threading)
    threading.Thread(target=main.one_and_done).start()

    return response
