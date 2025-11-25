import torch
import torch.nn as nn

class CenteringNet(nn.Module):
    """
    Predicts a 2D vector (x, y) from a 640x360 input image.
    Uses CoordConv to explicitly explicitly encode spatial position.
    """

    def __init__(self):
        super().__init__()
        
        # INPUT 3 channels (RGB) + 2 channels (X, Y coordinates) = 5
        self.enc1 = self.conv_block(5, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(128, 256)
        self.spatial_pool = nn.AdaptiveMaxPool2d((4, 7))

        # Flattened size: 256 channels * 4 height * 7 width = 7168 features
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7168, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2) # 2d vector output
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def append_coords(self, x):
        """
        Generates and appends (x, y) coordinate channels to the input tensor.
        x shape: (Batch, 3, H, W) -> Returns: (Batch, 5, H, W)
        """
        batch_size, _, height, width = x.shape
        y_coords = torch.linspace(-1, 1, height, device=x.device).view(1, 1, height, 1)
        x_coords = torch.linspace(-1, 1, width, device=x.device).view(1, 1, 1, width)
        y_channel = y_coords.expand(batch_size, 1, height, width)
        x_channel = x_coords.expand(batch_size, 1, height, width)
        
        # Concatenate along the channel dimension
        return torch.cat([x, x_channel, y_channel], dim=1)

    def forward(self, x):
        # Inject Coordinate Information
        x = self.append_coords(x)
        
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.bottleneck(x)
        x = self.spatial_pool(x)
        return self.regressor(x)