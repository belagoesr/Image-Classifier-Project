import argparse
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import utils

parser = argparse.ArgumentParser()

parser = utils.init_parser(parser)

results = parser.parse_args()

# args
path = results.path[0]
learning_rate, weight_decay, momentum = results.learning_rate, results.weight_decay, results.momentum
gpu, save_dir, hidden_units= results.gpu, results.save_dir, results.hidden_units
arch, epochs, dropout = results.arch, results.epochs, results.dropout

# load data from path
data_transforms, image_datasets, dataloaders, cat_to_name = utils.load_data(path)

# load pretrained model
model = models.__dict__[arch](pretrained=True)

# get pretrained model in_features number for the last layer
in_features, last_layer_name = utils.model_info(model)

# freeze pretrained model parameters
if hasattr(model, 'features'): 
    for param in model.features.parameters():
        param.requires_grad = False
else: # resnet
    for param in model.parameters():
        param.requires_grad = False

# create network with custom classifier
model = utils.create_network(model, in_features, last_layer_name, hidden_units, dropout)
print(model)

# set loss
criterion = nn.NLLLoss()

# set optimizer parameters
if hasattr(model, 'classifier'): 
    #optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD(model.classifier.parameters(), lr = learning_rate, 
    	momentum=momentum, weight_decay=weight_decay)

elif hasattr(model, 'fc'): # resnet
    #optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD(model.fc.parameters(), lr = learning_rate, 
    	momentum=momentum, weight_decay=weight_decay)

# set device to run
if(gpu):
	device = 'cuda'
else:
	device = 'cpu'

# print training parameters

# remove first and last elements to print hidden units only
layers = hidden_units[:-1][1:]

print('Training parameters:')
print('	arch = ',arch, '/ epochs =',epochs, '/ learning_rate =', learning_rate, '/ dropout =', dropout)
print('	hidden_units =', layers, '/ weight_decay =', weight_decay, '/ save_dir = ', save_dir)
print('	momentum:', momentum, '/ selected device:', device,'\n')

cuda_available = torch.cuda.is_available()

if (device=='cuda' and cuda_available) or (device=='cpu'): 

	if(device=='cpu' and cuda_available): print('Cuda available, better switch to gpu...')

	# move model to device
	model.to(device)
	print('training....')

	# train model
	utils.train(model, epochs, dataloaders, device, optimizer, criterion)

	# save checkpoint
	if(save_dir):
		save_path = save_dir + '/checkpoint_' + arch + '.pth'
	else:
		save_path = 'checkpoint_' + arch + '.pth'

	if hasattr(model, 'classifier'): 
	    utils.save_checkpoint(image_datasets, model.classifier, arch, model, optimizer, save_path)
	elif hasattr(model, 'fc'): # resnet
		utils.save_checkpoint(image_datasets, model.fc, arch, model, optimizer, save_path)   

else:
	print('Cuda not available')


