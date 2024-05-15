import torchvision
import torch.nn as nn
import torch


class myResnet(nn.Module):
    
    def __init__(self, orig_model):
        super(myResnet, self).__init__()
        self.orig = nn.Sequential(*(list(orig_model.children())[:-1]))
        for param in self.orig.parameters():
            param.requires_grad = True
        self.classifier  = nn.Sequential(
                              nn.Linear(512, 256),
                              nn.ReLU(),
                              nn.BatchNorm1d(256),
                              nn.Linear(256, 128),
                              nn.ReLU(),
                              nn.Linear(128, 1))

    def forward(self, x):

        x = self.orig(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return torch.sigmoid(logits)

