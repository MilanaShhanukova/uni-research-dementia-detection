import torch.nn as nn
import torch.nn.functional as F


class AttentionedModel(nn.Module):
    def __init__(self):
        super(AttentionedModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(5)
        self.conv2 = nn.Conv2d(128, 256, 3)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(5)
        self.multihead = nn.MultiheadAttention(256, 8, batch_first=True)
        self.linear_1 = nn.Linear(256 * 136, 256)
        self.linear_2 = nn.Linear(256, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = x.flatten(start_dim=2)
        x = x.permute(0, 2, 1)

        attentioned, _ = self.multihead(x, x, x)
        attentioned = attentioned.flatten(start_dim=1)
        result = self.linear_2(self.activation(self.linear_1(attentioned))).squeeze()
        return result