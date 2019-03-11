import torch
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import numpy as np
from PIL import Image
import argparse
from utils import load_data, model_info

def init_parser(parser):

    parser.add_argument('image_path', metavar='Image path', type=str, nargs='+', help='path for image')

    parser.add_argument('checkpoint', metavar='checkpoint', type=str, nargs='+', help='path for image')

    parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=1,
                        help='Return top K most likely classes')

    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default=None,
                        help='Use a mapping of categories to real names Ex:--category_names cat_to_name.json')

    parser.add_argument('--gpu', action='store_true', default=False, dest='gpu',
                        help='Use GPU for inference')

    return parser

def load_checkpoint(filepath, device):

    if device=='cpu':
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filepath)
        
    model = models.__dict__[checkpoint['pretrained']](pretrained=True)

    _, last_layer_name = model_info(model)

    if(last_layer_name == 'classifier'): 
        classifier = checkpoint['classifier']
        model.classifier = classifier

    elif(last_layer_name == 'fc'): 
        fc = checkpoint['classifier']
        model.fc = fc

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['dict']
    model.optimizer = checkpoint['optimizer']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = image
    w, h = im.size
    
    # resize the images where the shortest side is 256 pixels and keep ratio
    if w == h:
        new_w, new_h = 256, 256
        
    elif h < w:
        ratio = w/h
        new_w = 256*ratio
        new_h = 256
        
    elif w < h:
        ratio = h/w
        new_w = 256 
        new_h = 256*ratio
        
    im.thumbnail((new_w, new_h))
    
    # crop at center
    left = (new_w - 224)/2
    top = (new_h - 224)/2
    right = (new_w + 224)/2
    bottom = (new_h + 224)/2
    
    cropped_im = im.crop((left, top, right, bottom))
    
    # img to array and change color channels to between 0,1
    np_image = np.array(cropped_im) / 255
        
    # normalize
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_im = (np_image - means) / std
    final_im = normalized_im.transpose((2, 0, 1))

    return final_im

def predict(image_path, model, device, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()
    with torch.no_grad():
        # process the image and transform to tensor
        im_arr = process_image(Image.open(image_path))
        im_tensor = torch.from_numpy(im_arr)

        # get the prediction
        img = im_tensor.unsqueeze(0)
        img = img.type(torch.FloatTensor).to(device)
        log_ps = model(img)
        ps = torch.exp(log_ps)

        # predict the top 5 classes
        top_ps, top_classes = ps.topk(topk, dim=1)

        # find class mapping
        classes_to_idx = model.class_to_idx
        idx_to_class = {v: k for k, v in classes_to_idx.items()}

        # map the classes
        mapped_classes = []
        arr = top_classes.squeeze().tolist()
        
        if isinstance(arr, int): # topk=1
            arr = [arr]

        for idx in arr:
            mapped_classes.append(idx_to_class[idx])

        return top_ps, mapped_classes

def map_category_names(top_classes, category_names):
    ''' map classes to names according to file category_names
    '''

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    names = []
    for label in top_classes:
        names.append(cat_to_name[label])
    return names

def test_validate(model, device, optimizer, criterion):

    _, _, dataloaders, _ = load_data('./flowers')

    with torch.no_grad():
        accuracy = 0
        for images, labels in dataloaders['test']:

            images = images.to(device)
            labels = labels.to(device)

            logps = model(images)
            logps = logps.to(device)
            loss = criterion(logps, labels)

            ps = torch.exp(logps) 
            _, top_class = ps.topk(1, dim=1) 
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")