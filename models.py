import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        if out_channels % num_groups != 0:
            for g in [16, 8, 4, 2, 1]:
                if out_channels % g == 0:
                    num_groups = g
                    break
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class GridNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(input_channels, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d(2),
            ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256), nn.MaxPool2d(2),
            ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512), nn.MaxPool2d(2),
            ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 512),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout2d(0.5),
            nn.Conv2d(4096, 1024, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout2d(0.5),
            nn.Conv2d(1024, num_classes, kernel_size=1)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        feature_dim = 512
        projection_dim = 128
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )
        
    def forward(self, x, return_projection=True):
        representation = self.features(x)
        x = self.classifier(representation)
        x = self.global_avg_pool(x)
        logits = x.view(x.size(0), -1)
        
        if not return_projection:
            return logits
        
        rep_flat = self.global_avg_pool(representation).view(representation.size(0), -1)
        projection = self.projector(rep_flat)
        return logits, projection