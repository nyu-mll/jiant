"""
Utility functions used by all huggingface transformer embedders. 
"""
def correct_sent_indexing(sent, tokenizer_required, _unk_id, _pad_id, max_pos):
    """ Correct id difference between transformers and AllenNLP.
    The AllenNLP indexer adds'@@UNKNOWN@@' token as index 1, and '@@PADDING@@' as index 0

    args:
        sent: batch dictionary, in which
            sent[self.tokenizer_required]: <long> [batch_size, var_seq_len] input token IDs

    returns:
        ids: <long> [bath_size, var_seq_len] corrected token IDs
        input_mask: <long> [bath_size, var_seq_len] mask of input sequence
    """
    assert (
        tokenizer_required in sent
    ), "transformers cannot find correcpondingly tokenized input"
    ids = sent[tokenizer_required]

    input_mask = (ids != 0).long()
    pad_mask = (ids == 0).long()
    # map AllenNLP @@PADDING@@ to _pad_id in specific transformer vocab
    unk_mask = (ids == 1).long()
    # map AllenNLP @@UNKNOWN@@ to _unk_id in specific transformer vocab
    valid_mask = (ids > 1).long()
    # shift ordinary indexes by 2 to match pretrained token embedding indexes
    if _unk_id is not None:
        ids = (ids - 2) * valid_mask + _pad_id * pad_mask + _unk_id * unk_mask
    else:
        ids = (ids - 2) * valid_mask + _pad_id * pad_mask
        assert (
            unk_mask == 0
        ).all(), "out-of-vocabulary token found in the input, but _unk_id of transformers model is not specified"
    if max_pos is not None:
        assert (
            ids.size()[-1] <= max_pos
        ), "input length exceeds position embedding capacity, reduce max_seq_len"

    return ids, input_mask