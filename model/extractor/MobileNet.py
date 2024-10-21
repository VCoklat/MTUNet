'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    """
    MobileNet model definition.

    Attributes:
        cfg (list): Configuration list defining the architecture of the MobileNet. 
                    Each element can be an integer (number of output channels) or a tuple 
                    (number of output channels, stride).

    Methods:
        __init__(num_classes=10, remove_linear=False):
            Initializes the MobileNet model.
            
            Args:
                num_classes (int): Number of output classes for the final linear layer. Default is 10.
                remove_linear (bool): If True, removes the final linear layer. Default is False.
        
        _make_layers(in_planes):
            Creates the layers of the MobileNet based on the configuration list.
            
            Args:
                in_planes (int): Number of input channels for the first layer.
            
            Returns:
                nn.Sequential: A sequential container of the layers.
        
        forward(x, feature=False):
            Defines the forward pass of the MobileNet.
            
            Args:
                x (torch.Tensor): Input tensor.
                feature (bool): If True, returns both the features and the final output. Default is False.
            
            Returns:
                torch.Tensor or tuple: If feature is False, returns the final output tensor. 
                                       If feature is True, returns a tuple of (features, final output).
    """
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10, remove_linear=False):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        if remove_linear:
            self.linear = None
        else:
            self.linear = nn.Linear(1024, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if self.linear is None:
            if feature:
                return out, None
            else:
                return out
        if feature:
            out1 = self.linear(out)
            return out, out1
        out = self.linear(out)
        return out
