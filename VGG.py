import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.features = self.make_layers([64, 64, 'M', 128, 128, 'M', 256, 256,
                                          256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def make_layers(cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for layer in cfg:
            if layer == 'M':
                layers.extend([nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
            else:
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=3, padding=1)
                if batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(num_features=layer), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = layer

        return nn.Sequential(*layers)
