from PIL import Image
import io
from io import BytesIO
import requests
import pickle

def get_prediction(image_path):

    resp = requests.post(urls['my_model'], files={'file': open(image_path, 'rb')}).json()
    print('Probabilities: ', resp["Probabilities"])
    print('Answer: ', resp["Answer"])
    return resp["Probabilities"], resp["Answer"]


urls = {'my_model': "http://127.0.0.1/predict/"}
img = "test_image1.jpg"

get_prediction(img)
