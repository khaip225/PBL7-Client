import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 2)
        return x

class CNN14(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN14, self).__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.fc1(x))
        x = self.fc_audioset(x)
        return x

def freeze_model_blocks(model):
    # Đóng băng các layer đầu để học nhanh hơn và giữ đặc trưng âm thanh cơ bản
    for param in model.bn0.parameters():
        param.requires_grad = False
    for param in model.conv_block1.parameters():
        param.requires_grad = False
    for param in model.conv_block2.parameters():
        param.requires_grad = False
    for param in model.conv_block3.parameters():
        param.requires_grad = False
    return model