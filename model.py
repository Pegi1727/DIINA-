python
import torch
import torch.nn as nn
‌
class InhibitoryGate(nn.Module):
def __init__(self, in_channels, reduction_ratio=16):
super(InhibitoryGate, self).__init__()
self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
self.bn1 = nn.BatchNorm2d(in_channels // reduction_ratio)
self.relu = nn.ReLU(inplace=True)
self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
self.sigmoid = nn.Sigmoid()
‌
def forward(self, x):
out = self.relu(self.bn1(self.conv1(x)))
gate_weights = self.sigmoid(self.conv2(out))
return x * gate_weights, gate_weights
‌
class DIINA(nn.Module):
def __init__(self, num_classes=10, block_config=[6, 12, 8, 4], growth_rate=32):
super(DIINA, self).__init__()
self.features = nn.Sequential(
nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
nn.BatchNorm2d(64),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
num_features = 64
self.dense_blocks = nn.ModuleList()
self.gates = nn.ModuleList()
for num_layers in block_config:
self.gates.append(InhibitoryGate(num_features))
layers = []
for _ in range(num_layers):
layers.append(nn.Sequential(
nn.BatchNorm2d(num_features),
nn.ReLU(inplace=True),
nn.Conv2d(num_features, growth_rate, kernel_size=3, padding=1)
))
num_features += growth_rate
self.dense_blocks.append(nn.ModuleList(layers))
self.classifier = nn.Sequential(
nn.BatchNorm2d(num_features),
nn.ReLU(inplace=True),
nn.AdaptiveAvgPool2d((1, 1)),
nn.Flatten(),
nn.Linear(num_features, num_classes)
)
‌
def forward(self, x):
x = self.features(x)
concat_feats = x
for i in range(len(self.dense_blocks)):
gated_feats, _ = self.gates[i](concat_feats)
block_outputs = [gated_feats]
for layer in self.dense_blocks[i]:
current_input = torch.cat(block_outputs, 1)
new_features = layer(current_input)
block_outputs.append(new_features)
concat_feats = torch.cat(block_outputs, 1)
return self.classifier(concat_feats)
