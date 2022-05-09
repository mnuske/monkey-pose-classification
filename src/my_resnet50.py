# import torchvision.models as models
# from torchvision.models.resnet import ResNet, BasicBlock

# class MyResNet18(ResNet):
#     def __init__(self):
#         super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        
#     def forward(self, x):
#         # change forward here
#         x = self.conv1(x)
#         return x


# model = MyResNet18()
# # if you need pretrained weights
# model.load_state_dict(models.resnet18(pretrained=True).state_dict())



# ALTERNATIVE  



import torchvision.models as models
import torch
import torch.nn as nn
import os


def new_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    if not self.training:
        x = self.sigmoid(x)
        # x = (x/x.sum())

    return x


# define a resnet instance
resent = models.resnet50()

# get pretrained
os.environ['TORCH_HOME'] = 'models'
resent.load_state_dict(models.resnet50(pretrained=True).state_dict())

# changing output layer for my number of classes
num_ftrs = resent.fc.in_features
num_classes = 4
resent.fc = nn.Linear(num_ftrs, num_classes)

# add sigmoid layer
resent.sigmoid = nn.Sigmoid()

# add new_forward function to the resnet instance as a class method
bound_method = new_forward.__get__(resent, resent.__class__)
setattr(resent, 'forward', bound_method)

# save my altered model version
torch.save(resent, 'models/my_resnet50.pth')