# this model was tried but found to not perform as well as the simple convolutional net
# but it may still work better when more data is available, so I'm saving it here.


class SpatialSoftmax(nn.Module):
    """
    Computes the 'center of mass' for each feature map. 
    This translates high-level feature activations directly into (x, y) 
    coordinates, which is more robust for precision robotics tasks than 
    flattening into a dense layer.
    """
    def __init__(self, height, width, channel):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel

        # Create coordinate grids normalized from -1 to 1
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, self.width),
            torch.linspace(-1, 1, self.height),
            indexing='ij'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1))
        self.register_buffer('pos_y', pos_y.reshape(-1))

    def forward(self, feature_map):
        batch_size = feature_map.size(0)
        # Reshape to (B, C, H*W) and apply softmax over spatial dimensions
        probs = F.softmax(feature_map.view(batch_size, self.channel, -1), dim=-1)
        
        # Calculate expected (x, y) coordinates
        expected_x = torch.sum(probs * self.pos_x, dim=-1, keepdim=True)
        expected_y = torch.sum(probs * self.pos_y, dim=-1, keepdim=True)
        
        # Output is (Batch, Channel * 2) representing (x, y) per channel
        return torch.cat([expected_x, expected_y], dim=-1).view(batch_size, -1)

class CenteringNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load a pre-trained ResNet-18
        # ResNet expects 3 channels. We modify the first layer to accept 5 (RGB + XY Coords)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        original_conv = resnet.conv1
        self.encoder_stem = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy existing RGB weights to the first 3 channels, initialize others to zero
        with torch.no_grad():
            self.encoder_stem.weight[:, :3, :, :] = original_conv.weight
            self.encoder_stem.weight[:, 3:, :, :] = 0 
            
        # Compose the ResNet backbone without the final FC and pooling layers
        self.backbone = nn.Sequential(
            self.encoder_stem,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4  # Output: (B, 512, H/32, W/32) -> 384/32 = 12x12
        )

        # Spatial Softmax turns the 12x12x512 feature map into 512 (x,y) pairs
        self.spatial_softmax = SpatialSoftmax(12, 12, 512)
        
        # Input features for heads: 512 channels * 2 (x and y per channel) = 1024
        head_input = 1024
        
        self.head_vector = nn.Sequential(
            nn.Linear(head_input, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        
        self.head_valid = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.head_gripper = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.head_angle = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def append_coords(self, x):
        batch_size, _, height, width = x.shape
        y_coords = torch.linspace(-1, 1, height, device=x.device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = torch.linspace(-1, 1, width, device=x.device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return torch.cat([x, x_coords, y_coords], dim=1)

    def forward(self, x):
        x = self.append_coords(x)
        features = self.backbone(x)
        
        # Extract spatial features via softmax center-of-mass
        ssm_feats = self.spatial_softmax(features)
        
        return (
            self.head_vector(ssm_feats),
            self.head_valid(ssm_feats),
            self.head_gripper(ssm_feats),
            self.head_angle(ssm_feats) * math.pi
        )