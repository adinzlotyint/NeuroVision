import torch.nn as nn
import torch.nn.functional as F

#Simple CNN model to classify 10 classes on 32x32 RGB data
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # -> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64 x 32 x 32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> 128 x 32 x 32
        self.pool = nn.MaxPool2d(2, 2)  #Pooling, reducing img by half
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # CIFAR-10 – 10 klas

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # -> 32x16x16
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # -> 64x8x8
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # -> 128x4x4
        x = x.view(x.size(0), -1)  # flatten, size: batch x 2048
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
