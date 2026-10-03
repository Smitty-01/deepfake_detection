import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn.layer4.parameters():
            p.requires_grad = True

        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(
            2048, 256, batch_first=True, bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1)

        out, _ = self.lstm(feats)
        out = out.mean(dim=1)
        return self.classifier(out)
