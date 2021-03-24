import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import torch.optim as optim


torch.manual_seed(0)
np.random.seed(0)
BATCH_SIZE = 64
SIGMA=0

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    
# list all transformations
transform = transforms.Compose([transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                    AddGaussianNoise(0., SIGMA)])


# download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)



# dataiter = iter(testloader)
# images, labels = dataiter.next()
# # show images
# imshow(torchvision.utils.make_grid(images))
