# -*- coding: utf-8 -*-
"""
ikarosZX
"""

import torch
torch.backends.cudnn.benchmark=True                 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

model_dir = './data/rn101_classifier3.pth'
img_dir = './data/test.jpg'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,std)
                                      ])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=False)
        for param in self.model.parameters():
            #with torch.no_grad():
            param.requires_grad = False 
 
        self.model.fc = nn.Linear(self.model.fc.in_features,2,bias=False)
    def forward(self, x):
        x = self.model(x)
        return x

my_model = Model()
my_model.load_state_dict(torch.load(model_dir)) 
my_model = my_model.cuda()   #use gpu to run the model
my_model.eval()

img = Image.open(img_dir).convert('RGB')
#change the shape of image to fit in the model
img = data_transforms(img)  
img = img.unsqueeze(0)
print('img.shape:',img.shape)
    
outputs = my_model(img.cuda())
_, preds = torch.max(outputs.data,1)
#I use the gpu0 to run the model, so I should use '.cpu()' before using '.numpy()'
a = preds.data.cpu().numpy() 
print('preds:',img_name,' ',a)
