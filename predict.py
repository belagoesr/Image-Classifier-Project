import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from images_utils import load_checkpoint, predict, init_parser, map_category_names, test_validate
import argparse


parser = argparse.ArgumentParser()
parser = init_parser(parser)
results = parser.parse_args()

# get args
image_path, checkpoint, gpu = results.image_path[0], results.checkpoint[0], results.gpu
category_names, top_k = results.category_names, results.top_k

# check if inference should be done on gpu
cuda_available = torch.cuda.is_available()
if gpu and cuda_available:
    device = 'cuda'
else:
    if gpu: print('gpu not available, proceeding inference with cpu...')
    device = 'cpu'

# load modelåç
model = load_checkpoint(checkpoint, device)
#print(model)
model.to(device)

# predict classes for image
top_ps, top_classes = predict(image_path, model, device, top_k)
top_ps = top_ps.squeeze(0)

if category_names:
    names = map_category_names(top_classes, category_names)
    output = names
else:
    output = top_classes

for x, ps in zip(output, top_ps.tolist()):
    if(category_names): 
        print("Name: {0}  Probability {1:.2f}".format(x, ps))
    else:
        print("Class: {0}  Probability {1:.2f}".format(x, ps))
        
