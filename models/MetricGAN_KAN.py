"""
Generator and discriminator used in MetricGAN+KAN

Original Author: Szu-Wei Fu 2020
Adapted by: Yemin Mai 2024
"""

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from efficient_kan import KANLinear
# from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convs import KANConv2DLayer

import speechbrain as sb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xavier_init_layer(
    in_size, out_size=None, spec_norm=True, layer_type=nn.Linear, **kwargs
):
    "Create a layer with spectral norm, xavier uniform init and zero bias"
    if out_size is None:
        out_size = in_size

    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)

    # Perform initialization
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)

    return layer


def shifted_sigmoid(x):
    "Computes the shifted sigmoid."
    return 1.2 / (1 + torch.exp(-(1 / 1.6) * x))


class Learnable_sigmoid(nn.Module):
    """Implementation of a leanable sigmoid.

    Arguments
    ---------
    in_features : int
        Input dimensionality
    """

    def __init__(self, in_features=257):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True  # set requiresGrad to true!

        # self.scale = nn.Parameter(torch.ones(1))
        # self.scale.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        return 1.2 * torch.sigmoid(self.slope * x)


class EnhancementGenerator(nn.Module):
    """Simple LSTM for enhancement with custom initialization.

    Arguments
    ---------
    input_size : int
        Size of the input tensor's last dimension.
    hidden_size : int
        Number of neurons to use in the LSTM layers.
    num_layers : int
        Number of layers to use in the LSTM.
    dropout : int
        Fraction of neurons to drop during training.
    """

    def __init__(
        self,
        input_size=257,
        hidden_size=40,
        num_layers=2,
        # dropout=0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.activation = nn.LeakyReLU(negative_slope=0.3)

        # self.blstm = sb.nnet.RNN.LSTM(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=dropout,
        #     bidirectional=True,
        # )
        # """
        # Use orthogonal init for recurrent layers, xavier uniform for input layers
        # Bias is 0
        # """
        # for name, param in self.blstm.named_parameters():
        #     if "bias" in name:
        #         nn.init.zeros_(param)
        #     elif "weight_ih" in name:
        #         nn.init.xavier_uniform_(param)
        #     elif "weight_hh" in name:
        #         nn.init.orthogonal_(param)

        self.gru_cell_f = nn.ModuleList([nn.GRUCell(input_size, hidden_size, device=device),
            *(nn.GRUCell(hidden_size, hidden_size, device=device) for _ in range(self.num_layers - 1))
            ])
        self.gru_cell_b = nn.ModuleList([nn.GRUCell(input_size, hidden_size, device=device),
            *(nn.GRUCell(hidden_size, hidden_size, device=device) for _ in range(self.num_layers - 1))
            ])
        
        self.gru_linear = KANLinear(hidden_size * 2, hidden_size * 2)
        self.linear = KANLinear(hidden_size * 2, 257)

        self.Learnable_sigmoid = Learnable_sigmoid()

    def forward(self, x: torch.Tensor, lengths):
        """Processes the input tensor x and returns an output tensor."""
        batch_size = x.size(0)
        seq_lengths = x.size(1)

        ht_f = [torch.zeros(batch_size, self.hidden_size, device=device), *(None for _ in range(self.num_layers - 1))]
        ht_b = [torch.zeros(batch_size, self.hidden_size, device=device), *(None for _ in range(self.num_layers - 1))]

        out = torch.zeros(batch_size, seq_lengths, self.hidden_size * 2, device=device)

        for i in range(seq_lengths):
            ht_f[0] = self.gru_cell_f[0](x[:, i, :], ht_f[0])
            ht_b[0] = self.gru_cell_b[0](x[:, -1 - i, :], ht_b[0])
            for j in range(1, self.num_layers):
                ht_f[j] = self.gru_cell_f[j](ht_f[j - 1], ht_f[j])
                ht_b[j] = self.gru_cell_b[j](ht_b[j - 1], ht_b[j])
            out[:, i, :] = self.gru_linear(torch.concat((ht_f[-1], ht_b[-1]), 1))

        out = self.linear(out)
        out = self.Learnable_sigmoid(out)

        return out


class MetricDiscriminator(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

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
        # activation=nn.LeakyReLU,
    ):
        super().__init__()

        # self.activation = activation(negative_slope=0.3)

        self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)

        # Modifications

        self.conv1 = KANConv2DLayer(2, base_channels, kernel_size, base_activation=nn.SiLU)
        self.conv2 = KANConv2DLayer(base_channels, base_channels, kernel_size, base_activation=nn.SiLU)

        self.Linear1 = KANLinear(in_features=base_channels, out_features=1)

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        out = self.BN(x)

        out = self.conv1(out)
        out = self.conv2(out)

        out = torch.mean(out, (2, 3))

        out = self.Linear1(out)

        return out
