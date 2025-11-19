# src/models/model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class SmallResNetMultiHead(nn.Module):
    """ResNet18 backbone con dos 'heads' (emocion, genero)."""
    def __init__(self, n_emotions: int = 8, n_genders: int = 2, pretrained: bool = False):
        super().__init__()
        self.backbone = models.resnet18(weights=None) if not pretrained else models.resnet18(pretrained=True)
        # adapt first conv for 1 channel
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.emotion_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_emotions)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_genders)
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        return self.emotion_head(feats), self.gender_head(feats)

class SimpleCNN(nn.Module):
    """Modelo pequeño para debugging / comparar rendimiento DML."""
    def __init__(self, n_emotions: int = 8, n_genders: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.e_head = nn.Linear(256, n_emotions)
        self.g_head = nn.Linear(256, n_genders)

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        return self.e_head(z), self.g_head(z)
