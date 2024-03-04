import torch
from torch import nn, Tensor
from zeta.nn import FeedForward
import torch.nn.functional as F


def output_head(x: Tensor, dim: int):
    """
    Applies a linear transformation followed by softmax activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim).
        dim (int): Dimension of the input tensor.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, dim) after applying linear transformation and softmax activation.
    """
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
        filter: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_mult = mlp_mult
        self.dropout = dropout
        self.heads = heads
        self.filter = filter

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
            # swish=True,
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

