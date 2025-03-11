import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import math

class ModifiedResNet50(nn.Module):
    def __init__(self, num_input_channels=4, num_classes=10, init_type='mean'):
        """
        Args:
            num_input_channels (int): Number of input channels
            num_classes (int): Number of output classes
            init_type (str): Initialization type for new channels. Options:
                - 'mean': Mean of existing channels
                - 'zero': Zero initialization
                - 'ones': Ones initialization
                - 'random_uniform': Uniform random initialization
                - 'random_normal': Normal random initialization
                - 'kaiming_uniform': Kaiming uniform initialization
                - 'kaiming_normal': Kaiming normal initialization
                - 'xavier_uniform': Xavier uniform initialization
                - 'xavier_normal': Xavier normal initialization
                - 'copy_first': Copy the first channel
                - 'copy_random': Copy a random channel
        """
        super(ModifiedResNet50, self).__init__()
        
        # Load pretrained model
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        pretrained_conv1_weights = base_model.conv1.weight.data
        
        # Create new conv layer
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        
        # Initialize new weights based on initialization type
        with torch.no_grad():
            new_weights = torch.zeros(64, num_input_channels, 7, 7)
            
            # Copy original channels
            new_weights[:, :3, :, :] = pretrained_conv1_weights
            
            # Initialize additional channels
            if num_input_channels > 3:
                if init_type == 'mean':
                    # Initialize as mean of RGB channels
                    new_weights[:, 3:, :, :] = pretrained_conv1_weights.mean(dim=1, keepdim=True)
                
                elif init_type == 'zero':
                    # Leave as zeros
                    pass
                
                elif init_type == 'ones':
                    # Initialize with ones
                    new_weights[:, 3:, :, :] = 1.0
                
                elif init_type == 'random_uniform':
                    # Initialize with uniform random values in [-1, 1]
                    rand_channels = torch.rand(64, num_input_channels-3, 7, 7) * 2 - 1
                    new_weights[:, 3:, :, :] = rand_channels
                
                elif init_type == 'random_normal':
                    # Initialize with normal distribution (μ=0, σ=0.02)
                    rand_channels = torch.randn(64, num_input_channels-3, 7, 7) * 0.02
                    new_weights[:, 3:, :, :] = rand_channels
                
                elif init_type == 'kaiming_uniform':
                    # Kaiming uniform initialization
                    fan_in = 7 * 7 * 3  # kernel_size * kernel_size * in_channels
                    bound = math.sqrt(6. / fan_in)
                    rand_channels = torch.rand(64, num_input_channels-3, 7, 7).uniform_(-bound, bound)
                    new_weights[:, 3:, :, :] = rand_channels
                
                elif init_type == 'kaiming_normal':
                    # Kaiming normal initialization
                    fan_in = 7 * 7 * 3
                    std = math.sqrt(2. / fan_in)
                    rand_channels = torch.randn(64, num_input_channels-3, 7, 7) * std
                    new_weights[:, 3:, :, :] = rand_channels
                
                elif init_type == 'xavier_uniform':
                    # Xavier uniform initialization
                    fan_in = 7 * 7 * 3
                    fan_out = 64 * 7 * 7
                    bound = math.sqrt(6. / (fan_in + fan_out))
                    rand_channels = torch.rand(64, num_input_channels-3, 7, 7).uniform_(-bound, bound)
                    new_weights[:, 3:, :, :] = rand_channels
                
                elif init_type == 'xavier_normal':
                    # Xavier normal initialization
                    fan_in = 7 * 7 * 3
                    fan_out = 64 * 7 * 7
                    std = math.sqrt(2. / (fan_in + fan_out))
                    rand_channels = torch.randn(64, num_input_channels-3, 7, 7) * std
                    new_weights[:, 3:, :, :] = rand_channels
                
                elif init_type == 'copy_first':
                    # Copy the first channel
                    new_weights[:, 3:, :, :] = pretrained_conv1_weights[:, 0:1, :, :]
                
                elif init_type == 'copy_random':
                    # Copy a random channel for each new channel
                    for i in range(3, num_input_channels):
                        random_channel = torch.randint(0, 3, (1,))
                        new_weights[:, i:i+1, :, :] = pretrained_conv1_weights[:, random_channel, :, :]
                
                else:
                    raise ValueError(f"Unknown initialization type: {init_type}")
            
            self.conv1.weight.data = new_weights

        # Copy the remaining initial layers
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        # Copy the first layer group (conv2_x)
        self.layer1 = base_model.layer1
        
        # Copy the second layer group (conv3_x)
        self.layer2 = base_model.layer2
        
        # Create partial third layer (up to conv4_block3_2_conv)
        self.partial_layer3 = nn.Sequential(
            base_model.layer3[0],  # conv4_block1
            base_model.layer3[1],  # conv4_block2
            nn.Sequential(
                base_model.layer3[2].conv1,    # conv4_block3_1_conv
                base_model.layer3[2].bn1,      # conv4_block3_1_bn
                base_model.layer3[2].relu,     # conv4_block3_1_relu
                base_model.layer3[2].conv2     # conv4_block3_2_conv
            )
        )
        
        # Add final 1x1 convolution layer for classification
        self.conv_final = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        # Initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.partial_layer3(x)
        
        # Final convolution
        x = self.conv_final(x)
        
        return x


import torch
import torch.nn as nn

class ResNet50Partial(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50Partial, self).__init__()
        
        # Initial convolution block (conv1)
        self.conv1_pad = nn.ZeroPad2d(3)
        self.conv1_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, bias=False)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_relu = nn.ReLU(inplace=True)
        
        # Pooling
        self.pool1_pad = nn.ZeroPad2d(1)
        self.pool1_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # conv2_block1
        self.conv2_block1_1_conv = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv2_block1_1_bn = nn.BatchNorm2d(64)
        self.conv2_block1_1_relu = nn.ReLU(inplace=True)
        self.conv2_block1_2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv2_block1_2_bn = nn.BatchNorm2d(64)
        self.conv2_block1_2_relu = nn.ReLU(inplace=True)
        self.conv2_block1_3_conv = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.conv2_block1_3_bn = nn.BatchNorm2d(256)
        self.conv2_block1_0_conv = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.conv2_block1_0_bn = nn.BatchNorm2d(256)
        self.conv2_block1_out = nn.ReLU(inplace=True)
        
        # conv2_block2
        self.conv2_block2_1_conv = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.conv2_block2_1_bn = nn.BatchNorm2d(64)
        self.conv2_block2_1_relu = nn.ReLU(inplace=True)
        self.conv2_block2_2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv2_block2_2_bn = nn.BatchNorm2d(64)
        self.conv2_block2_2_relu = nn.ReLU(inplace=True)
        self.conv2_block2_3_conv = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.conv2_block2_3_bn = nn.BatchNorm2d(256)
        self.conv2_block2_out = nn.ReLU(inplace=True)
        
        # conv2_block3
        self.conv2_block3_1_conv = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.conv2_block3_1_bn = nn.BatchNorm2d(64)
        self.conv2_block3_1_relu = nn.ReLU(inplace=True)
        self.conv2_block3_2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv2_block3_2_bn = nn.BatchNorm2d(64)
        self.conv2_block3_2_relu = nn.ReLU(inplace=True)
        self.conv2_block3_3_conv = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.conv2_block3_3_bn = nn.BatchNorm2d(256)
        self.conv2_block3_out = nn.ReLU(inplace=True)

        # conv3_block1
        self.conv3_block1_1_conv = nn.Conv2d(256, 128, kernel_size=1, stride=2, bias=False)
        self.conv3_block1_1_bn = nn.BatchNorm2d(128)
        self.conv3_block1_1_relu = nn.ReLU(inplace=True)
        self.conv3_block1_2_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv3_block1_2_bn = nn.BatchNorm2d(128)
        self.conv3_block1_2_relu = nn.ReLU(inplace=True)
        self.conv3_block1_3_conv = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.conv3_block1_3_bn = nn.BatchNorm2d(512)
        self.conv3_block1_0_conv = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.conv3_block1_0_bn = nn.BatchNorm2d(512)
        self.conv3_block1_out = nn.ReLU(inplace=True)

        # conv3_block2
        self.conv3_block2_1_conv = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv3_block2_1_bn = nn.BatchNorm2d(128)
        self.conv3_block2_1_relu = nn.ReLU(inplace=True)
        self.conv3_block2_2_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv3_block2_2_bn = nn.BatchNorm2d(128)
        self.conv3_block2_2_relu = nn.ReLU(inplace=True)
        self.conv3_block2_3_conv = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.conv3_block2_3_bn = nn.BatchNorm2d(512)
        self.conv3_block2_out = nn.ReLU(inplace=True)

        # conv3_block3
        self.conv3_block3_1_conv = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv3_block3_1_bn = nn.BatchNorm2d(128)
        self.conv3_block3_1_relu = nn.ReLU(inplace=True)
        self.conv3_block3_2_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv3_block3_2_bn = nn.BatchNorm2d(128)
        self.conv3_block3_2_relu = nn.ReLU(inplace=True)
        self.conv3_block3_3_conv = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.conv3_block3_3_bn = nn.BatchNorm2d(512)
        self.conv3_block3_out = nn.ReLU(inplace=True)

        # conv3_block4
        self.conv3_block4_1_conv = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv3_block4_1_bn = nn.BatchNorm2d(128)
        self.conv3_block4_1_relu = nn.ReLU(inplace=True)
        self.conv3_block4_2_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv3_block4_2_bn = nn.BatchNorm2d(128)
        self.conv3_block4_2_relu = nn.ReLU(inplace=True)
        self.conv3_block4_3_conv = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.conv3_block4_3_bn = nn.BatchNorm2d(512)
        self.conv3_block4_out = nn.ReLU(inplace=True)

        # conv4_block1
        self.conv4_block1_1_conv = nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False)
        self.conv4_block1_1_bn = nn.BatchNorm2d(256)
        self.conv4_block1_1_relu = nn.ReLU(inplace=True)
        self.conv4_block1_2_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv4_block1_2_bn = nn.BatchNorm2d(256)
        self.conv4_block1_2_relu = nn.ReLU(inplace=True)
        self.conv4_block1_3_conv = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.conv4_block1_3_bn = nn.BatchNorm2d(1024)
        self.conv4_block1_0_conv = nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.conv4_block1_0_bn = nn.BatchNorm2d(1024)
        self.conv4_block1_out = nn.ReLU(inplace=True)

        # conv4_block2
        self.conv4_block2_1_conv = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_block2_1_bn = nn.BatchNorm2d(256)
        self.conv4_block2_1_relu = nn.ReLU(inplace=True)
        self.conv4_block2_2_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv4_block2_2_bn = nn.BatchNorm2d(256)
        self.conv4_block2_2_relu = nn.ReLU(inplace=True)
        self.conv4_block2_3_conv = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.conv4_block2_3_bn = nn.BatchNorm2d(1024)
        self.conv4_block2_out = nn.ReLU(inplace=True)

        # conv4_block3 (up to conv4_block3_2_conv)
        self.conv4_block3_1_conv = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv4_block3_1_bn = nn.BatchNorm2d(256)
        self.conv4_block3_1_relu = nn.ReLU(inplace=True)
        self.conv4_block3_2_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        # final conv layer that map 256 to num_classes
        self.conv_final = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)



    def forward(self, x):
        # Initial convolution block
        x = self.conv1_pad(x)
        x = self.conv1_conv(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        
        # Pooling
        x = self.pool1_pad(x)
        x = self.pool1_pool(x)
        
        # conv2_block1
        identity = x
        x = self.conv2_block1_1_conv(x)
        x = self.conv2_block1_1_bn(x)
        x = self.conv2_block1_1_relu(x)
        x = self.conv2_block1_2_conv(x)
        x = self.conv2_block1_2_bn(x)
        x = self.conv2_block1_2_relu(x)
        x = self.conv2_block1_3_conv(x)
        x = self.conv2_block1_3_bn(x)
        identity = self.conv2_block1_0_conv(identity)
        identity = self.conv2_block1_0_bn(identity)
        x += identity
        x = self.conv2_block1_out(x)
        
        # conv2_block2
        identity = x
        x = self.conv2_block2_1_conv(x)
        x = self.conv2_block2_1_bn(x)
        x = self.conv2_block2_1_relu(x)
        x = self.conv2_block2_2_conv(x)
        x = self.conv2_block2_2_bn(x)
        x = self.conv2_block2_2_relu(x)
        x = self.conv2_block2_3_conv(x)
        x = self.conv2_block2_3_bn(x)
        x += identity
        x = self.conv2_block2_out(x)
        
        # conv2_block3
        identity = x
        x = self.conv2_block3_1_conv(x)
        x = self.conv2_block3_1_bn(x)
        x = self.conv2_block3_1_relu(x)
        x = self.conv2_block3_2_conv(x)
        x = self.conv2_block3_2_bn(x)
        x = self.conv2_block3_2_relu(x)
        x = self.conv2_block3_3_conv(x)
        x = self.conv2_block3_3_bn(x)
        x += identity
        x = self.conv2_block3_out(x)

        # conv3_block1
        identity = x
        x = self.conv3_block1_1_conv(x)
        x = self.conv3_block1_1_bn(x)
        x = self.conv3_block1_1_relu(x)
        x = self.conv3_block1_2_conv(x)
        x = self.conv3_block1_2_bn(x)
        x = self.conv3_block1_2_relu(x)
        x = self.conv3_block1_3_conv(x)
        x = self.conv3_block1_3_bn(x)
        identity = self.conv3_block1_0_conv(identity)
        identity = self.conv3_block1_0_bn(identity)
        x += identity
        x = self.conv3_block1_out(x)

        # conv3_block2
        identity = x
        x = self.conv3_block2_1_conv(x)
        x = self.conv3_block2_1_bn(x)
        x = self.conv3_block2_1_relu(x)
        x = self.conv3_block2_2_conv(x)
        x = self.conv3_block2_2_bn(x)
        x = self.conv3_block2_2_relu(x)
        x = self.conv3_block2_3_conv(x)
        x = self.conv3_block2_3_bn(x)
        x += identity
        x = self.conv3_block2_out(x)

        # conv3_block3
        identity = x
        x = self.conv3_block3_1_conv(x)
        x = self.conv3_block3_1_bn(x)
        x = self.conv3_block3_1_relu(x)
        x = self.conv3_block3_2_conv(x)
        x = self.conv3_block3_2_bn(x)
        x = self.conv3_block3_2_relu(x)
        x = self.conv3_block3_3_conv(x)
        x = self.conv3_block3_3_bn(x)
        x += identity
        x = self.conv3_block3_out(x)

        # conv3_block4
        identity = x
        x = self.conv3_block4_1_conv(x)
        x = self.conv3_block4_1_bn(x)
        x = self.conv3_block4_1_relu(x)
        x = self.conv3_block4_2_conv(x)
        x = self.conv3_block4_2_bn(x)
        x = self.conv3_block4_2_relu(x)
        x = self.conv3_block4_3_conv(x)
        x = self.conv3_block4_3_bn(x)
        x += identity
        x = self.conv3_block4_out(x)

        # conv4_block1
        identity = x
        x = self.conv4_block1_1_conv(x)
        x = self.conv4_block1_1_bn(x)
        x = self.conv4_block1_1_relu(x)
        x = self.conv4_block1_2_conv(x)
        x = self.conv4_block1_2_bn(x)
        x = self.conv4_block1_2_relu(x)
        x = self.conv4_block1_3_conv(x)
        x = self.conv4_block1_3_bn(x)
        identity = self.conv4_block1_0_conv(identity)
        identity = self.conv4_block1_0_bn(identity)
        x += identity
        x = self.conv4_block1_out(x)

        # conv4_block2
        identity = x
        x = self.conv4_block2_1_conv(x)
        x = self.conv4_block2_1_bn(x)
        x = self.conv4_block2_1_relu(x)
        x = self.conv4_block2_2_conv(x)
        x = self.conv4_block2_2_bn(x)
        x = self.conv4_block2_2_relu(x)
        x = self.conv4_block2_3_conv(x)
        x = self.conv4_block2_3_bn(x)
        x += identity
        x = self.conv4_block2_out(x)

        # conv4_block3 (up to conv4_block3_2_conv)
        identity = x
        x = self.conv4_block3_1_conv(x)
        x = self.conv4_block3_1_bn(x)
        x = self.conv4_block3_1_relu(x)
        x = self.conv4_block3_2_conv(x)

        x = self.conv_final(x)
        
        return x

if __name__ == '__main__':
    model = ResNet50Partial()
    x = torch.randn(1, 4, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, 256, 14, 14]