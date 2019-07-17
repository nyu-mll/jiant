import torch


def get_seg_ids(ids, sep_id):
    """ Dynamically build the segment IDs for a concatenated pair of sentences
    Searches for index SEP_ID in the tensor. Supports BERT or XLNet-style padding.

    args:
        ids (torch.LongTensor): batch of token IDs

    returns:
        seg_ids (torch.LongTensor): batch of segment IDs

    example:
    > sents = ["[CLS]", "I", "am", "a", "cat", ".", "[SEP]", "You", "like", "cats", "?", "[SEP]"]
    > token_tensor = torch.Tensor([[vocab[w] for w in sent]]) # a tensor of token indices
    > seg_ids = _get_seg_ids(token_tensor, sep_id=102) # BERT [SEP] ID
    > assert seg_ids == torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    """
    sep_idxs = (ids == sep_id).nonzero()[:, 1]
    seg_ids = torch.ones_like(ids)
    for row, idx in zip(seg_ids, sep_idxs[::2]):
        row[: idx + 1].fill_(0)

    torch.set_printoptions(threshold=5000)
    print(ids)
    print(seg_ids)
    print(sep_id)
    print()
    return seg_ids
