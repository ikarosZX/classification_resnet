# -*- coding: utf-8 -*-
import torch
torch.backends.cudnn.benchmark=True                 

import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from torch.optim import lr_scheduler
 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
 
total_epochs = 1             #you can change it to any number but not 0 #here I just use 1 to do some tests
lr = 0.01
delay_lr = 0.1
delay_epoch_lr = 15          #every 15 epochs, the lr would be lr * delay_lr 
save_dir = './data/test.pth' #save the model to your dir
data_dir = './data'          #the data for train and test
bs = 32                      #batch_size
'''
there are three folders under [data], which are [train], [val], [test].
and the images should be classifiered by given the different names to the their folder,
for example, the images of cat should be put into the floder named cat.
'''
#load the data 
def load_dataset(data_dir,bs):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),  # change the shape of the image
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    }
    
    data_dir2 = data_dir
    image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir2, x),
					  data_transforms[x])for x in ['train', 'val','test']}
    
    #print('classes:',datasets.ImageFolder(os.path.join(data_dir, 'val'),data_transforms['val']).classes) #classes: ['down', 'up']
    #print(datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val']).class_to_idx)      #{'down': 0, 'up': 1}

    data_loaders = {x: data.DataLoader(image_datasets[x],
                    batch_size=bs,num_workers=12, shuffle=True)
                    for x in ['train', 'val','test']}
    
    data_size = {x : len(image_datasets[x]) for x in ['train', 'val','test']}
    return data_loaders, data_size
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
	#choose one of [resnet18,resnet34,resnet50,resnet101,resnet152]
        self.model = models.resnet101(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False 
        self.model.fc = nn.Linear(self.model.fc.in_features,2,bias=False)
    def forward(self, x):
        x = self.model(x)
        return x
 
 
def train(data_loader, data_size,delay_epoch_lr,delay_lr):
	model = Model()
	model = model.cuda()  # training with GPU
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.model.parameters(), lr=lr, momentum=0.9)
	scheduler1 = lr_scheduler.StepLR(optimizer, step_size=delay_epoch_lr, gamma=delay_lr)  
 
	#---------finetune-----------#
	for epoch in range(total_epochs):
		tqdm.write('Epoch {}/{}'.format(epoch, total_epochs-1))
 
		#-------- training ----------#
		for mode in ['train', 'val']:
			if mode == 'train':
				scheduler1.step()  
				model.train=True
				tot_loss = 0.0
			else:
				model.train = False
			sum_right = 0
			for data1 in tqdm(data_loader[mode]):
				inputs,labels = data1
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
				if mode == 'train':
					optimizer.zero_grad()
				outputs = model(inputs)   
 
				_, preds = torch.max(outputs.data,1)
				#print(preds)
				if mode == 'train':
					loss = criterion(outputs, labels)
					loss.backward()  
					optimizer.step()   
					tot_loss += loss.data
				sum_right += torch.sum(preds == labels.data).to(torch.float32)
        
			if mode == 'train':
				epoch_loss = tot_loss/data_size[mode]
				print('train loss: ', epoch_loss)
			epoch_acc = sum_right/data_size[mode]
			print(mode + ' acc: ', epoch_acc)
		torch.save(model.state_dict(),save_dir)
	return model
 
def test(data_loader, data_size, model):
    model.train = False
    sum_right = 0
    for data2 in tqdm(data_loader['test']):
        inputs,labels = data2
        #print('inputs:',type(inputs))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        #print('outputs:',outputs.data)
        _, preds = torch.max(outputs.data,1)
        sum_right += torch.sum(preds == labels.data).to(torch.float32)
    print('test acc: ', sum_right/data_size['test'])
 
def my_run():
    data_loader, data_size = load_dataset(data_dir,bs)
    model = train(data_loader,data_size,delay_epoch_lr,delay_lr)
    test(data_loader, data_size, model)
 
 
if __name__ == '__main__':
    my_run()
