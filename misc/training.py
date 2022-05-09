import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image
from torchvision.models import resnet18, resnet50
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
from torch.optim import lr_scheduler, SGD
import torch.nn as nn
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import copy

from pathlib import Path
import logging
import time


from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
exit()
writer = SummaryWriter()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



os.environ['TORCH_HOME'] = 'models'
model = resnet50(pretrained=True)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


transforming_data = {
    'train': transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]),
    'validation': transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]),
    'test': transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
    ])
}


directory_data = Path('/media/hdd2/matthias', 'datasets', 'monkey_poses')


datasets_images = {x: datasets.ImageFolder((directory_data / x), transforming_data[x])
                        for x in ['train', 'validation']
                    }


loaders_data = {x: torch.utils.data.DataLoader(datasets_images[x], batch_size=64,
                                             shuffle=True, num_workers=16)
                    for x in ['train', 'validation']
                }


# images, labels = next(iter(loaders_data['train']))
# grid = torchvision.utils.make_grid(images)
# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)



sizes_datasets = {x: len(datasets_images[x]) for x in ['train', 'validation']}

class_names = datasets_images['train'].classes



plt.ion()   # This is the interactive mode
def visualize_data(input, title=None):
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    if title is not None:
       plt.title(title)
    plt.pause(0.001)  ## Here we are pausing a bit so that plots are updated
inputs_data, classes = next(iter(loaders_data['train']))
## This is the code for getting a batch of training data
out = torchvision.utils.make_grid(inputs_data)
## Here we are making a grid from batch
visualize_data(out, title=[class_names[x] for x in classes])




def model_training(res_model, criterion, optimizer, scheduler, number_epochs=25):
    since = time.time()
    best_resmodel_wts = copy.deepcopy(res_model.state_dict())
    best_accuracy = 0.0
    for epochs in trange(number_epochs):
        # print('Epoch {}/{}'.format(epochs, number_epochs - 1))
        # print('-' * 10)
        for phase in ['train', 'validation']: ## Here each epoch is having a training and validation phase
            if phase == 'train':
               res_model.train()  ## Here we are setting our model to training mode
            else:
               res_model.eval()   ## Here we are setting our model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loaders_data[phase]: ## Iterating over data.
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() ## here we are making the gradients to zero

                with torch.set_grad_enabled(phase == 'train'): ## forwarding and then tracking the history if only in train
                    outputs = res_model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train': # backward and then optimizing only if it is in training phase
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.StepLR(optimizer, step_size=5, gamma=0.1)# changed from scheduler.step(), -> do lr step every 5 epochs, i think??

            writer.add_scalar('Learn Rate', 0.1**(epochs//5+1), epochs)

            epoch_loss = running_loss / sizes_datasets[phase]
            epoch_acc = running_corrects.double() / sizes_datasets[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'validation' and epoch_acc > best_accuracy: ## deep copy the model
                best_accuracy = epoch_acc
                best_resmodel_wts = copy.deepcopy(res_model.state_dict())


            if phase == 'validation':
                writer.add_scalar('Loss/validation', epoch_loss, epochs)
                writer.add_scalar('Accuracy/validation', epoch_acc, epochs)
            else:
                writer.add_scalar('Loss/train', epoch_loss, epochs)
                writer.add_scalar('Accuracy/train', epoch_acc, epochs)

        print()
        if epochs % 5 == 0:# save checkpoints
            torch.save(res_model, 'model_{:d}.pth'.format(epochs))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_accuracy))

    # load best model weights
    res_model.load_state_dict(best_resmodel_wts)
    return res_model



finetune_model = resnet50(pretrained=True)
num_ftrs = finetune_model.fc.in_features

finetune_model.fc = nn.Linear(num_ftrs, 4) #output features needs the same dimension as input features??? @laura

finetune_model = finetune_model.to(device)

criterion = nn.CrossEntropyLoss()

finetune_optim = SGD(finetune_model.parameters(), lr=0.1, momentum=0.9)



finetune_model = model_training(finetune_model, criterion, finetune_optim, lr_scheduler,
                       number_epochs=50)



torch.save(finetune_model, 'model_final.pth')



writer.close()