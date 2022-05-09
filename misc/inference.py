import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from torchvision.models import resnet18, resnet50
import torchvision.transforms as transforms
from torchvision import datasets

import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



from pathlib import Path



from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()




os.environ['TORCH_HOME'] = 'models'
# model = resnet50(pretrained=True)



res_model = torch.load('model_final.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
res_model.to(device)




directory_data = Path('/media/hdd2/matthias', 'datasets', 'monkey_poses')


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



datasets_images = {'test': datasets.ImageFolder((directory_data / 'test'), transforming_data['test'])}
loaders_data = {'test': torch.utils.data.DataLoader(datasets_images['test'], batch_size=62,
                                             shuffle=True, num_workers=16)}
                    

size_dataset = {'test': len(datasets_images['test'])}
class_names = datasets_images['test'].classes

gt_labels = np.empty((1,0))
pred_labels = np.empty((1,0))
with torch.no_grad:
    for inputs, labels in loaders_data['test']: ## Iterating over data.
        inputs = inputs.to(device)
        # print(gt_labels.shape, np.array(labels).reshape(1,-1).shape)
        gt_labels = np.concatenate((gt_labels, np.array(labels).reshape(1,-1)), axis=1)
        
        outputs = res_model(inputs)
        _, preds = torch.max(outputs, 1)
        pred_labels = np.concatenate((pred_labels, np.array(preds.cpu()).reshape(1,-1)), axis=1)

# gt = np.array(gt_labels).reshape(-1)
# pred = np.array(pred_labels).reshape(-1)
gt = gt_labels.flatten().astype(int)
pred = pred_labels.flatten().astype(int)

print(class_names)
from sklearn.metrics import confusion_matrix

np.save('gt.npy', gt)
np.save('pred.npy', pred)
cmat = confusion_matrix(gt, pred)
import matplotlib.pyplot as plt
plt.imshow(cmat, cmap='plasma')
plt.colorbar()
plt.xticks(labels, np.arrange(len(labels)))
plt.show()



def model_visualization(res_model, num_images=6):
    was_training = res_model.training
    res_model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loaders_data['validation']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = res_model(inputs)
            _, preds = torch.max(outputs, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            visualize_data(inputs.cpu().data[j])

            if images_so_far == num_images:
               res_model.train(mode=was_training)
               return
        res_model.train(mode=was_training)