import torch
import torch.nn as nn
from core.registry import register_model
from DIINA_14_research import DIINA14

@register_model('diina')
class DIINAClassifier(nn.Module):
    def __init__(self, vocab_size=20, num_classes=10):
        super().__init__()

        try:
            self.backbone = DIINA14(vocab_size=vocab_size)
        except:
            self.backbone = DIINA14()

        self.embed = getattr(self.backbone, 'embed', None)
        self.blocks = getattr(self.backbone, 'blocks', [])
        self.norm = getattr(self.backbone, 'final_norm', nn.Identity())

        d_model = 256
        if self.embed is not None:
            try:
                d_model = self.embed.embedding_dim
            except:
                pass

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)

        for block in self.blocks:
            try:
                x, _ = block(x)
            except:
                x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.head(x)
