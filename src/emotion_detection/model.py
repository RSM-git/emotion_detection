import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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


    def make_layers(self):
        pass


    def forward(self, x):
        x = self.input(x)
        x = self.layers(x)

        return self.mlp(x)

if __name__ == '__main__':
    model = EmotionNet()
    test_tensor = torch.randn(8, 1, 48, 48)
    model(test_tensor)
