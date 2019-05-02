"""
ResNet block and masked-softmax used in PRPN
Reference: Parsing-Reading-Predict Networks (PRPN; Shen et al., 2018)
All the modules in this file are taken without change from: https://github.com/yikangshen/PRPN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def stick_breaking(logits):
    e = F.sigmoid(logits)
    z = (1 - e).cumprod(dim=1)
    p = torch.cat([e.narrow(1, 0, 1), e[:, 1:] * z[:, :-1]], dim=1)

    return p


def softmax(x, mask=None):
    """
     softmax function with masking for self-attention
    """
    max_x, _ = x.max(dim=-1, keepdim=True)
    e_x = torch.exp(x - max_x)
    if not (mask is None):
        e_x = e_x * mask
    out = e_x / (e_x.sum(dim=-1, keepdim=True) + 1e-8)

    return out


class ResBlock(nn.Module):
    """
     Resnet block used in parsing network of PRPN
    """

    def __init__(self, ninp, nout, dropout, nlayers=1):
        super(ResBlock, self).__init__()

        self.nlayers = nlayers

        self.drop = nn.Dropout(dropout)

        self.res = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(ninp, ninp),
                    nn.BatchNorm1d(ninp),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ninp, ninp),
                    nn.BatchNorm1d(ninp),
                )
                for _ in range(nlayers)
            ]
        )

    def forward(self, input):
        # input = self.drop(input)
        for i in range(self.nlayers):
            input = F.relu(self.res[i](input) + input)
        return input
