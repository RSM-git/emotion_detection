import torch.nn as nn
import torch

class EmotionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 x 48 x 48 images
        self.input = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
        )

        self.layers = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.Dropout2d(0.2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
        )

        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 6 * 6, 256),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.layers(x)

        return self.mlp(x)

if __name__ == '__main__':
    model = EmotionNet()
    test_tensor = torch.randn(8, 1, 48, 48)
    model(test_tensor)
