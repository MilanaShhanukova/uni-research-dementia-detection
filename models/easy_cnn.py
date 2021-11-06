import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)])

        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(12672, 1)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, data):
        x = data.unsqueeze(1)

        for conv in self.convs:
            x = conv(x)
            x = self.pool(self.activation(x))

        x = self.flatten(x)
        logits = self.linear(x)
        results = self.softmax(logits)
        return results.flatten()
