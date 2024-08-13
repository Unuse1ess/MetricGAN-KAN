"""
Generator g2

Original Author: Szu-Wei Fu 2020
Adapted by: Yemin Mai 2024
"""

from torch import nn
import speechbrain as sb

from efficient_kan import KANLinear
from models.utils import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The same as g1 but some hyper-params, but here g2 is still preserved
class EnhancementGenerator(nn.Module):
    """
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
        num_layers=1,
        dropout=0,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.3)

        self.blstm = sb.nnet.RNN.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0
        """
        for name, param in self.blstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        self.linear1 = KANLinear(2 * hidden_size, 257)

        self.Learnable_sigmoid = Learnable_sigmoid()

    def forward(self, x, lengths):
        """Processes the input tensor x and returns an output tensor."""
        out, _ = self.blstm(x, lengths=lengths)

        out = self.linear1(out)
        out = self.Learnable_sigmoid(out)

        return out
