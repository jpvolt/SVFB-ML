#written by João Pedro de Carvalho Voltani 
import os
import numpy as np  
import torch 
import torch.nn as nn #imports neural net module from pytorch
from torch.utils.data import Dataset, DataLoader
import cv2 #inports opencv
import pandas as pd
import math
import sys
import argparse

device = None #device used for training, initialized un svfb_trainner.py
dataset_location = '' # dataset path , initialized un svfb_trainner.py
l_r = 0.0025 #lerarning rate, initialized un svfb_trainner.py
num_epochs = 50 # number of epochs, initialized un svfb_trainner.py
batch_size = 350 #batch size for training  =  number of imgs training simutaneous =  mini batch size, initialized un svfb_trainner.py
shuffle = False #Shufle dataset , initialized un svfb_trainner.py
modelname = '' #model save name, initialized un svfb_trainner.py
cnn = None #object variable, initialized un svfb_trainner.py



class InceptionModule(nn.Module): #conv net module, inspired on google inception archtecture
	def __init__(self, in_channels, a, b, c, d):
		super(InceptionModule,self).__init__()

		self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=a, kernel_size=1,stride=1)

		self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=b, kernel_size=3,stride=1, padding=1)

		self.conv5x5 = nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=5,stride=1, padding=2)

		self.maxPool = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
			nn.Conv2d(in_channels=in_channels, out_channels=d, kernel_size=1,stride=1))


	def forward(self,input_data):

		l1 = self.conv1x1(input_data)
		l2 = self.conv3x3(input_data)
		l3 = self.conv5x5(input_data)
		l4 = self.maxPool(input_data)
		out = torch.cat((l1,l2,l3,l4), dim=1)

		return out #out = axbx33

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet,self).__init__()
		self.l1 = nn.Sequential(
			nn.BatchNorm2d(1),
			InceptionModule(1,7,7,3,1),
			nn.ReLU())

		self.l2 = nn.Sequential(
			nn.BatchNorm2d(18),
			InceptionModule(18,7,7,3,1),
			nn.ReLU())

		self.l3 = nn.Sequential(#residual
			nn.BatchNorm2d(18),
			InceptionModule(18,7,7,3,1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))

		
		self.l4 = nn.Sequential(
			nn.BatchNorm2d(18),
			InceptionModule(18,7,7,3,1),
			nn.ReLU())

		self.l5 = nn.Sequential(
			nn.BatchNorm2d(18),
			InceptionModule(18,7,7,3,1),
			nn.ReLU())

		self.l6 = nn.Sequential(#residual
			nn.BatchNorm2d(18),
			InceptionModule(18,7,7,3,1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))


		self.linear = nn.Linear(594,500)
		self.linear2 = nn.Linear(500,2)
	def forward(self, img):
		out1 = self.l1(img)
		out2 = self.l2(out1)
		out3 = self.l3(out2)
		out4 = self.l4(out3)
		out5 = self.l5(out4)
		out6 = self.l6(out5)
	
		out6_linear = out6.reshape(out6.size(0), -1)
		l1 = self.linear(out6_linear)
		out = self.linear2(l1)

		return out


def saveModel():
	torch.save(cnn, 'models/'+modelname)


def train():
	
	criterion =  nn.CrossEntropyLoss() #loss
	optimizer = torch.optim.Adagrad(cnn.parameters(), lr = l_r)
	
	dataset = svfbDataset(dataset_location=dataset_location)
	training_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

	cycle = 0
	for batch_imgs, batch_labels in training_dataloader:
	
		
		batch_imgs, batch_labels = torch.tensor(batch_imgs).to(device), torch.tensor(batch_labels).to(device)
			
		for epoch in range(num_epochs):
			
			outputs = cnn(batch_imgs)#feed forward
			loss = criterion(outputs, batch_labels)#compute the loss
			optimizer.zero_grad()#reset gradients
			loss.backward()#backward pass
			optimizer.step()
	
			print('cycle: ',cycle,' loss: {}'.format(loss.item()))
		cycle = cycle + 1

	saveModel()
	

class svfbDataset(Dataset):
	
	def __init__(self, dataset_location):
		
		self.sessions = None
		for _ ,_ , files in os.walk(dataset_location):
			for file in files:
	
				session = np.load(str(dataset_location+'/'+file))
				if self.sessions is None:
					self.sessions = session
				else:
					self.sessions = np.concatenate((self.sessions, session), axis=0)

		

	def __len__(self):
		return self.sessions.shape[0]

	def __getitem__(self, idx):
		image = self.sessions[idx][0]
		image = torch.from_numpy(image.astype(float))
		image = image.unsqueeze(0)
		image = image.permute(0,2,1)
		if self.sessions[idx][1] == 1:
			labels = torch.tensor(1).long()
		else:
			labels = torch.tensor(0).long()
		
		

		return image.type('torch.FloatTensor'), torch.tensor(labels).type('torch.LongTensor')