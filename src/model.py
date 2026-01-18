import torch.nn as nn
import torch

class EmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 x 48 x 48 images
        self.input = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
        )

        self.layers = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3),
                nn.BatchNorm2d(32)
        )

        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8192, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 7)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.layers(x)

        return self.mlp(x)

if __name__ == '__main__':
    model = EmotionNet()
    test_tensor = torch.randn(8, 1, 48, 48)
    model(test_tensor)
