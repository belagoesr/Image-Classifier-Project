import json
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session

def init_parser(parser):
	parser.add_argument('path', metavar='Images path', type=str, nargs='+', help='path for images')

	parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16',
	                    help='Pretrained model architecture Ex: vgg16, densenet201, ...')

	parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default = 0.001,
	                    help='model hyperparameter')

	parser.add_argument('--momentum', action='store', dest='momentum', type=float, default = 0.9,
	                    help='model hyperparameter')

	parser.add_argument('--weight_decay', action='store', dest='weight_decay', type=float, default = 0.0,
	                    help='model hyperparameter')

	parser.add_argument('--epochs', action='store', dest='epochs', type=int, default = 5,
	                    help='model hyperparameter.')

	parser.add_argument('--dropout', action='store', dest='dropout', type=float, default = 0.25,
	                    help='model hyperparameter.')

	parser.add_argument('--hidden_units', action='store', dest='hidden_units',nargs='*', type=int,default = [1028],
	                    help='model hyperparameter. Ex: --hidden_units 512 (one hidden layer) or --hidden_units 1028 512 (multiple hidden layers)')

	parser.add_argument('--gpu', action='store_true', default=False, dest='gpu',
	                    help='Train model in a gpu')

	parser.add_argument('--save_dir', action='store', dest='save_dir',
	                    help='Directory to save checkpoint')
	return parser

def load_data(path):

	data_dir = path
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	input_size = 224
	normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]);
	data_transforms = {
	    'train': transforms.Compose([transforms.RandomRotation(30),
	                               transforms.RandomResizedCrop(input_size),
	                               transforms.RandomHorizontalFlip(),
	                               transforms.ToTensor(),
	                               normalize
	                               ]),
	    
	    'valid': transforms.Compose([transforms.Resize(256),
	                               transforms.CenterCrop(input_size),
	                               transforms.ToTensor(),
	                               normalize
	                                ]),
	    
	    'test':  transforms.Compose([transforms.Resize(256),
	                                transforms.CenterCrop(input_size),
	                                transforms.ToTensor(),
	                                normalize
	                                ]),
	}

	image_datasets = {
	    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
	    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
	    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
	}

	dataloaders = {
	    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
	    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
	    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
	}

	with open('cat_to_name.json', 'r') as f:
	    cat_to_name = json.load(f)

	return data_transforms, image_datasets, dataloaders, cat_to_name

def model_info(model):
    
    last_layer = list(model.children())[-1:][0]
    last_layer_name = list(model.state_dict())[-1:][0].split('.')[0]

    if type(last_layer) == torch.nn.modules.container.Sequential:

        for layer in last_layer:

            if(type(layer) == torch.nn.modules.linear.Linear):
                in_features = layer.in_features
                break;

            elif(type(layer)==torch.nn.modules.conv.Conv2d):
                in_features = layer.in_channels
                break;
    else:

        in_features = list(model.children())[-1:][0].in_features
    return in_features, last_layer_name

def create_network(model, in_features, last_layer_name, hidden_units, dropout):
    
    # freeze pretrained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # create hidden layers
    hidden_units.insert(0,in_features)
    hidden_units.append(102)
    n = len(hidden_units)
    layers = []
    for i in range(n-2):
        layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))

    # create output layer
    layers.append(nn.Linear(hidden_units[-2], hidden_units[-1]))
    layers.append(nn.LogSoftmax(dim=1))
    
    classifier = nn.Sequential(*layers)

    # replace last layer by classifier
    if last_layer_name == 'classifier':
        model.classifier = classifier
        
    elif last_layer_name == 'fc':
        model.fc = classifier
    
    return model

def calculate_accuracy(logps, labels):
    
    ps = torch.exp(logps) 
    _, top_class = ps.topk(1, dim=1) 
    equals = top_class == labels.view(*top_class.shape)
    
    return torch.mean(equals.type(torch.FloatTensor)).item()

def validate(model, criterion, dataloaders, device):
    
    valid_len = len(dataloaders['valid'])
    
    with torch.no_grad():
        
        accuracy = 0
        valid_loss = 0

        # switch to evaluation mode
        model.eval()
        
        for images, labels in dataloaders['valid']:

            images = images.to(device)
            labels = labels.to(device)

            logps = model(images)
            logps = logps.to(device)
            loss = criterion(logps, labels)
            valid_loss += loss.item()

            accuracy += calculate_accuracy(logps, labels)
            
        acc = accuracy/valid_len
        v_loss = valid_loss/valid_len
        
    return acc, v_loss
           
def train(model, epochs, dataloaders, device, optimizer, criterion):

    model.to(device)
    epochs = epochs;
    steps = 0;
    print_step = 15;
    running_loss = 0
    
    with active_session():
        
        for epoch in range(epochs):
            
            for images, labels in dataloaders['train']:
                
                #switch to train mode
                model.train()
                steps += 1

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logps = model(images)
                logps = logps.to(device)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if(steps % print_step == 0):

                    validation_acc, valid_loss = validate(model, criterion, dataloaders, device)

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_step:.3f}.. "
                          f"Valid loss: {valid_loss:.3f}.. "
                          f"Valid accuracy: {validation_acc:.3f}")

                    running_loss = 0

def save_checkpoint(image_datasets, classifier, arch, model, optimizer, save_path):
	checkpoint = {
	    'dict': image_datasets['train'].class_to_idx,
	    'pretrained': arch,
	    'classifier': classifier,
	    'state_dict': model.state_dict(),
	    'optimizer': optimizer.state_dict(),
	}	
	torch.save(checkpoint, save_path);
