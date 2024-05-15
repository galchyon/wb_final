import requests
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from io import BytesIO
from model import myResnet
import torchvision.models as models
import pickle

class Prediction:
    
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")
        model_orig = models.resnet18()
        model = myResnet(model_orig)
        model.to(device)    
        model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.model = model
        self.device = device

    def predict(self, image):

        img_pil = Image.open(BytesIO(image))
        img_tensor = self.preprocess(img_pil)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            prediction = self.model(img_tensor).item()
            if prediction >= 0.5:
                ans = 1
            else:
                ans = 0
        outs = {'Probabilities': prediction,
                'Answer': ans}
        return outs
