import torch
import torch.nn as nn
from core.registry import register_model

@register_model('mamba')
class MambaLike(nn.Module):
    def __init__(self, d_model=256, num_classes=10):
        super().__init__()
        self.embed = nn.Embedding(20, d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)
        return self.fc(x.mean(dim=1))
