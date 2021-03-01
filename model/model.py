import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from base import BaseModel

class TBModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 2))
    def forward(self):        
        return self.model