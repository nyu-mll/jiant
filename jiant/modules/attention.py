import torch
from torch import nn
from overrides import overrides
from allennlp.modules.attention import Attention


@Attention.register("Bahdanau")
class BahdanauAttention(Attention):
    def __init__(
        self, tensor_1_dim: int, tensor_2_dim: int, att_hid_size: int = 100, normalize: bool = True
    ) -> None:
        super().__init__(normalize)

        self.linear_1 = nn.Linear(tensor_1_dim + tensor_2_dim, att_hid_size)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(att_hid_size, 1)

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        vector_large = vector.unsqueeze(1).expand([-1, matrix.shape[1], -1])
        matrix_vector = torch.cat([matrix, vector_large], dim=2)
        forward_1 = self.tanh(self.linear_1(matrix_vector))
        forward_2 = self.linear_2(forward_1)

        return forward_2.squeeze(2)
