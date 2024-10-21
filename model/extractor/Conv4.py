from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Conv4(nn.Module):
    """
    A convolutional neural network module with four convolutional layers followed by a linear layer.

    Args:
        num_classes (int): The number of output classes for the linear layer.
        drop_dim (bool, optional): If True, flattens the output of the last convolutional layer before passing it to the linear layer. Default is True.
        extract (bool, optional): If True, returns both the output of the linear layer and the flattened output of the last convolutional layer. Default is False.

    Attributes:
        drop_dim (bool): Whether to flatten the output of the last convolutional layer.
        extract (bool): Whether to return both the output of the linear layer and the flattened output of the last convolutional layer.
        conv1 (nn.Module): The first convolutional block.
        conv2 (nn.Module): The second convolutional block.
        conv3 (nn.Module): The third convolutional block.
        conv4 (nn.Module): The fourth convolutional block.
        linear (nn.Module): The linear layer.

    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor from the linear layer.
                If `extract` is True, also returns the flattened output of the last convolutional layer.
    """
    def __init__(self, num_classes, drop_dim=True, extract=False):
        super(Conv4, self).__init__()
        self.drop_dim = drop_dim
        self.extract = extract
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.linear = nn.Linear(1600, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_f = self.conv4(x)

        if self.drop_dim:
            x_f = x_f.view(x_f.size(0), -1)
        x_out = self.linear(x_f)

        if self.extract:
            return x_out, x_f
        else:
            return x_out
