import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, inp_dim=256, out_dim=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp_dim, out_dim//2, 1)
        self.conv2 = nn.Conv2d(out_dim//2, out_dim//2, 3, padding=1)
        self.conv3 = nn.Conv2d(out_dim//2, out_dim, 1)
        self.skip_conv = nn.Conv2d(inp_dim, out_dim, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_dim//2)
        self.bn2 = nn.BatchNorm2d(out_dim//2)
        self.bn3 = nn.BatchNorm2d(out_dim)
    def forward(self, x):
        res = self.skip_conv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return out+res
    
class HourGlass(nn.Module):
    def __init__(self, n=5):
        super(HourGlass, self).__init__()
        self.n = n
        self.skip_branch = ResBlock()
        self.pool = nn.MaxPool2d(2)
        if self.n>1: self.block1 = HourGlass(self.n-1)
        else: self.block1 = ResBlock()
        self.block2 = ResBlock()
        self.block3 = ResBlock()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        res = self.skip_branch(x)
        out = self.up(self.block3(self.block2(self.block1(self.pool(x)))))
        return res+out
    
class Last(nn.Module):
    def __init__(self):
        super(Last, self).__init__()
        self.last1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),
            nn.Sigmoid()
        )
        self.last2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1)
        )
        self.last3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1)
        )
    def forward(self, x):
        return [self.last1(x), self.last2(x), self.last3(x)]
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 128),
            nn.MaxPool2d(2),
            ResBlock(128, 128),
            ResBlock(128, 256),
            nn.MaxPool2d(2)
        )
        self.hourglass1 = HourGlass()
        self.hourglass2 = HourGlass()
        self.last = Last()
    def forward(self, img):
        return self.last(self.hourglass2(self.hourglass1(self.preprocess(img))))
