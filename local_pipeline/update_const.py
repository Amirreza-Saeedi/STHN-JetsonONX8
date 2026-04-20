import torch
import torch.nn as nn
from utils import *


class CNN_64(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, init_dim=164):
        super(CNN_64, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(init_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = input_dim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = input_dim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer5 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        outputdim_final = outputdim
        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x1 = self.layer10(x)
        return x1

class CNN(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(164, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim_final = outputdim

        ### global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)

        return x

class GMA(nn.Module):
    def __init__(self, args, sz):
        super().__init__()
        self.args = args

        if args.corr_level == 2:
            init_dim = 164
        elif args.corr_level == 4:
            init_dim = 326
        self.cnn = CNN_64(128, init_dim=init_dim)

    def forward(self, corr, flow):      
        delta_flow = self.cnn(torch.cat((corr, flow), dim=1))
        return delta_flow