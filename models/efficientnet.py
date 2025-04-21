import torch
import torch.nn as nn
import timm

class EfficientNet(nn.Module):

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)

        self.feature_dim = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):

        features = self.backbone(x)

        logits = self.head(features)

        return logits
