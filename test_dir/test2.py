import requests


def get_prediction(image_path):

    resp = requests.post(urls['my_model'], files={'file': open(image_path, 'rb')}).json()
    print('Probabilities: ', resp["Probabilities"])
    print('Answer: ', resp["Answer"])
    return resp["Probabilities"], resp["Answer"]


urls = {'my_model': "http://127.0.0.1/predict/"}
#img = "test_image1.jpg"
print('enter image path')
img = input()

get_prediction(img)
