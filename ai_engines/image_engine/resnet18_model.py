import torch
import torch.nn as nn
from torchvision import models

def ResNet18(num_classes=1, in_channels=3):
    """
    ResNet18 for binary classification (Normal vs Pneumonia)
    
    Args:
        num_classes: Number of output classes (default=1 for binary classification with sigmoid)
        in_channels: Number of input channels (default=3 for RGB images)
    
    Returns:
        model: ResNet18 model with custom FC layer
    
    Note: This model expects:
        - Input shape: (batch_size, 3, 224, 224)
        - Output shape: (batch_size, 1) 
        - Loss function: BCEWithLogitsLoss
        - Activation in inference: sigmoid
    """
    model = models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # FC layer structure: Dropout + Linear (trained with BCEWithLogitsLoss)
    # Output: 1 neuron for binary classification (sigmoid applied during inference)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)  # 1 output for binary classification
    )
    return model

def freeze_model_blocks(model):
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    return model
