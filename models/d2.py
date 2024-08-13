"""
Discriminator d1

Original Author: Szu-Wei Fu 2020
Adapted by: Yemin Mai 2024
"""

import torch
from torch import nn

from efficient_kan import KANLinear
from kan_convs import KANConv2DLayer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetricDiscriminator(nn.Module):
    """Metric estimator for enhancement training.

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    activation : Callable
        Function to apply between layers.
    """

    def __init__(
        self,
        kernel_size=(5, 5),
        base_channels=15,
        num_layers=3
    ):
        super().__init__()

        self.num_layers = num_layers

        self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)
        
        self.convs = nn.Sequential([
            KANConv2DLayer(2, base_channels, kernel_size, base_activation=nn.SiLU),
            *[KANConv2DLayer(base_channels, base_channels, kernel_size, base_activation=nn.SiLU) for _ in range(num_layers - 1)]
        ])

        self.Linear1 = KANLinear(in_features=base_channels, out_features=1)

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        out = self.BN(x)

        out = self.convs(out)

        out = torch.mean(out, (2, 3))

        out = self.Linear1(out)

        return out
