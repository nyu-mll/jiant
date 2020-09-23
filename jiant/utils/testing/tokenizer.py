from typing import List

from jiant.utils.python.datastructures import BiMap
from jiant.tasks.core import FeaturizationSpec


class SimpleSpaceTokenizer:

    pad_token = "<pad>"
    cls_token = "<cls>"
    sep_token = "<sep>"
    unk_token = "<unk>"
    SPECIAL_TOKENS = [pad_token, cls_token, sep_token, unk_token]

    def __init__(self, vocabulary: List[str], add_special=True):
        if add_special:
            vocabulary = self.SPECIAL_TOKENS + vocabulary
        self.tokens_to_ids, self.ids_to_tokens = BiMap(
            a=vocabulary, b=list(range(len(vocabulary)))
        ).get_maps()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.tokens_to_ids[token] for token in tokens]

    def tokenize(self, string: str) -> List[str]:
        return [
            token if token in self.tokens_to_ids else self.unk_token for token in string.split()
        ]

    @classmethod
    def get_feat_spec(cls, max_seq_length: int) -> FeaturizationSpec:
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=0,
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=1,
            sep_token_extra=False,
        )
