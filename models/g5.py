"""
Generator g3

Original Author: Szu-Wei Fu 2020
Adapted by: Yemin Mai 2024
"""

from torch import nn
import speechbrain as sb

from efficient_kan import KANLinear
from models.utils import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnhancementGenerator(nn.Module):
    """Simple GRU-KAN for enhancement with custom initialization.

    Arguments
    ---------
    input_size : int
        Size of the input tensor's last dimension.
    hidden_size : int
        Number of neurons to use in the GRU-KAN layers.
    num_layers : int
        Number of layers to use in the GRU-KAN.
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

        # self.activation = nn.LeakyReLU(negative_slope=0.3)

        self.gru_cell_f = nn.ModuleList([
                nn.GRUCell(input_size, hidden_size),
                *(nn.GRUCell(hidden_size, hidden_size) for _ in range(self.num_layers - 1))
            ])
        self.gru_cell_b = nn.ModuleList([
                nn.GRUCell(input_size, hidden_size),
                *(nn.GRUCell(hidden_size, hidden_size) for _ in range(self.num_layers - 1))
            ])
        
        self.gru_linear = KANLinear(hidden_size * 2, hidden_size * 2)
        self.linear = KANLinear(hidden_size * 2, 257)

        self.Learnable_sigmoid = Learnable_sigmoid()

    def forward(self, x: torch.Tensor, lengths):
        """Processes the input tensor x and returns an output tensor."""
        device = x.device
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
