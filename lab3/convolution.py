import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        raise NotImplementedError("You need to implement this")

    def forward(self, x):
        """CNN forward
        Args:
            x (torch.Tensor):
                [B, S, F] Batch size x sequence length x feature size
                padded inputs
        Returns:
            torch.Tensor: [B, O] Batch size x CNN output size cnn outputs
        """
        raise NotImplementedError("You need to implement this")
