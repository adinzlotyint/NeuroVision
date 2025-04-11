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

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        # After applying four pooling layers: 32 -> 16 -> 8 -> 4 -> 2 (spatial dimensions)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 channels -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64 channels -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 128 channels -> 4x4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 256 channels -> 2x2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        # Input size: 3*32*32 = 3072 features
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x