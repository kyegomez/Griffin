import torch
import torch.nn.functional as F
from torch import Tensor, nn
from zeta.nn import FeedForward


class RGLRU(nn.Module):
    """
    Real-Gated Linear Recurrent Unit (RG-LRU) for 3D input tensors.
    """

    def __init__(self, dim, mult: int):
        super().__init__()
        self.dim = dim
        hidden_dim = dim * mult
        self.hidden_dim = hidden_dim
        self.c = 8  # Scalar-valued constant

        # Initialize weights
        self.Wa = nn.Parameter(torch.Tensor(hidden_dim, dim))
        self.Wx = nn.Parameter(torch.Tensor(hidden_dim, dim))
        self.ba = nn.Parameter(torch.Tensor(hidden_dim))
        self.bx = nn.Parameter(torch.Tensor(hidden_dim))
        self.Lambda = nn.Parameter(torch.Tensor(hidden_dim))  # Λ

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.Wa, mode="fan_in", nonlinearity="linear"
        )
        nn.init.kaiming_normal_(
            self.Wx, mode="fan_in", nonlinearity="linear"
        )
        nn.init.constant_(self.ba, 0)
        nn.init.constant_(self.bx, 0)
        # Initialize Λ such that a is between 0.9 and 0.999
        self.Lambda.data.uniform_(
            torch.logit(torch.tensor(0.9)),
            torch.logit(torch.tensor(0.999)),
        )

    def forward(self, x):
        """
        Forward pass for sequences.

        Parameters:
        - x (Tensor): Input tensor with shape (batch_size, sequence_length, dim)

        Returns:
        - y (Tensor): Output tensor with shape (batch_size, sequence_length, hidden_dim)
        """
        batch_size, sequence_length, _ = x.shape
        ht = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        y = []

        for t in range(sequence_length):
            xt = x[:, t, :]
            rt = torch.sigmoid(torch.matmul(xt, self.Wa) + self.ba)
            it = torch.sigmoid(torch.matmul(xt, self.Wx) + self.bx)
            a = torch.sigmoid(self.Lambda)
            at = a / self.c**rt
            ht = at * ht + ((1 - at**2) ** 0.5) * (it * xt)
            y.append(ht.unsqueeze(1))

        y = torch.cat(y, dim=1)
        return y


def output_head(x: Tensor, dim: int):
    """
    Applies a linear transformation followed by softmax activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim).
        dim (int): Dimension of the input tensor.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, dim) after applying linear transformation and softmax activation.
    """
    x = nn.LayerNorm(dim)(x)

    # Linear transformation
    x = nn.Linear(dim, dim)(x)

    # Softmax
    return F.softmax(x, dim=-1)


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) module.

    Args:
        dim (int): The dimension of the input tensor.

    Attributes:
        scale (float): The scaling factor for the normalized output.
        g (nn.Parameter): The learnable parameter used for scaling.

    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized output tensor.

        """
        return F.normalize(x, dim=-1) * self.scale * self.g


class GriffinResidualBlock(nn.Module):
    """
    GriffinResidualBlock is a residual block used in the Griffin model.

    Args:
        dim (int): The input dimension.
        depth (int): The depth of the block.
        mlp_mult (int): The multiplier for the hidden dimension in the feedforward network.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        heads (int, optional): The number of attention heads. Defaults to 8.
        filter (int, optional): The filter size for the convolutional layer. Defaults to 4.

    Attributes:
        dim (int): The input dimension.
        depth (int): The depth of the block.
        mlp_mult (int): The multiplier for the hidden dimension in the feedforward network.
        dropout (float): The dropout rate.
        heads (int): The number of attention heads.
        filter (int): The filter size for the convolutional layer.
        norm (RMSNorm): The normalization layer.
        mlp (FeedForward): The feedforward network.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        mlp_mult: int,
        dropout: float = 0.1,
        heads: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        self.heads = heads

        # Norm
        self.norm = RMSNorm(dim)

        # Feedforward
        self.mlp = FeedForward(
            dim,
            dim,
            mlp_mult,
            post_act_ln=True,
            dropout=dropout,
            *args,
            **kwargs,
        )

        # RG-LRU
        self.lru = RGLRU(
            dim,
            mult=4,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the GriffinResidualBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        b, s, d = x.shape

        skip = x

        # Norm
        x = self.norm(x)
        print(x.shape)

        # Temporal Mixing Block
        linear_1, linear_2 = nn.Linear(d, d)(x), nn.Linear(d, d)(x)
        print(linear_1.shape, linear_2.shape)

        # Conv1d
        linear_1 = nn.Conv1d(
            in_channels=s,
            out_channels=s,
            kernel_size=3,
            padding=1,
        )(linear_1)
        print(linear_1.shape)

        # RG-LRU
        # linear_1 = self.lru(linear_1)

        # Gelu on linear 2
        linear_2 = nn.GELU()(linear_2)

        # Element wise multiplication to merge the paths
        x = linear_1 * linear_2
        print(x.shape)

        # skip
        x += skip

        # Skip2
        skip2 = x

        # Norm
        x = self.norm(x)

        # Feedforward
        x = self.mlp(x)

        return x + skip2


class Griffin(nn.Module):
    """
    Griffin module for performing Griffin Residual Network operations.

    Args:
        dim (int): Dimension of the input tensor.
        depth (int): Number of residual blocks in the network.
        mlp_mult (int): Multiplier for the hidden dimension of the MLP layers.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        heads (int, optional): Number of attention heads. Defaults to 8.
        filter (int, optional): Filter size for the convolutional layers. Defaults to 4.

    Attributes:
        dim (int): Dimension of the input tensor.
        depth (int): Number of residual blocks in the network.
        mlp_mult (int): Multiplier for the hidden dimension of the MLP layers.
        dropout (float): Dropout probability.
        heads (int): Number of attention heads.
        filter (int): Filter size for the convolutional layers.
        layers (nn.ModuleList): List of GriffinResidualBlock layers.

    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        seq_len: int,
        depth: int = 8,
        mlp_mult: int = 4,
        dropout: float = 0.1,
        heads: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.depth = depth
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        self.heads = heads

        # Layers
        self.layers = nn.ModuleList()

        # Add layers
        self.layers.append(
            GriffinResidualBlock(
                dim,
                depth,
                mlp_mult,
                dropout,
                heads,
                *args,
                **kwargs,
            )
        )

        # Embedding layer
        self.emb = nn.Embedding(
            num_tokens,
            dim,
        )

        # Rmsnorm
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Griffin module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.

        """
        x = self.emb(x)
        x = self.norm(x)

        # Loop
        for layer in self.layers:
            x = layer(x) + x

        return output_head(x, self.dim)
