'''
ResNet in PyTorch.

This implementation is based on kuangliu's code
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and the PyTorch reference implementation
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Explainer(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_channels, mask):
        super(Explainer, self).__init__()
        self.in_planes = 64

        # Input conv.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks.
        channels = 64
        stride = 1
        blocks = []
        for num in num_blocks:
            blocks.append(self._make_layer(block, channels, num, stride=stride))
            channels *= 2
            stride = 2
        self.layers = nn.ModuleList(blocks)

        # Output layer.
        self.num_classes = num_classes
        # if num_classes is not None:
        #     self.linear = nn.Linear(512*block.expansion, num_classes)
        # print(block.expansion)
        self.conv = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1)

        self.upconv = nn.ConvTranspose2d(
            num_classes, 
            num_classes, 
            kernel_size=4,  # Even kernel size for upscaling
            stride=2,       # Scale factor of 2
            padding=1,
            bias=False
        )
        self.upconv2 = nn.ConvTranspose2d(
            num_classes, 
            num_classes, 
            kernel_size=4,  # Even kernel size for upscaling
            stride=2,       # Scale factor of 2
            padding=1,
            bias=False
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input conv.
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks.
        for i,layer in enumerate(self.layers):
            out = layer(out)
            # print(f"out {i}", out.shape)

        # Add these lines:
        # out = self.conv(out)                    # Conv2d with number of classes
        # out = out.reshape(out.size(0), out.size(1), -1)  # Reshape
        # out = out.permute(0, 2, 1)             # Permute dimensions
        # print("out1", out.shape)
        out = self.conv(out)
        # print("out2", out.shape)
        out = self.upconv(out)
        # print("out3", out.shape)
        out = self.upconv2(out)
        # print("out4", out.shape)
        return out

        # Output layer.
        # if self.num_classes is not None:
        #     out = F.avg_pool2d(out, 4)
        #     out = out.view(out.size(0), -1)
        #     out = self.linear(out)

        # return out


def Explainer18(num_classes, in_channels=3, mask=[14,14]):
    return Explainer(BasicBlock, [2, 2, 2, 2], num_classes, in_channels, mask)


def Explainer34(num_classes, in_channels=3, mask=[14,14]):
    return Explainer(BasicBlock, [3, 4, 6, 3], num_classes, in_channels, mask)


def Explainer50(num_classes, in_channels=3, mask=[14,14]):
    return Explainer(Bottleneck, [3, 4, 6, 3], num_classes, in_channels, mask)
