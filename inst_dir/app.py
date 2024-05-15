import fastapi
from fastapi import FastAPI, File, UploadFile
import pickle
import numpy as np
import predictor
import uvicorn
import urllib.request

app = FastAPI()

model = pickle.load(urllib.request.urlopen("https://drive.google.com/uc?export=download&id=1RT64ELoGb947rlyQj7TxQaZ0IJRCimt7"))

@app.get("/")
def read_root():
       return {"message": "Welcome to the Spam-Classifier"}

@app.post("/predict/")
def predict(file: UploadFile = File(...)):
       
       res = model.predict(file.file.read())
       
       return res

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
