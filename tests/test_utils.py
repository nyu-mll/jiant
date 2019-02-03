import src.utils.utils as utils
import unittest
from io import StringIO

def test_load_tsv_has_labels_one_sentence():
    mock_csv = StringIO("sentence	label",
    "it 's a charming and often affecting journey . 	1",
    "unflinchingly bleak and desperate 	0")
    with open('temp_output.tsv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(mock_csv)
    max_seq_len = 30
    data = utils.load_tsv('data_output.tsv', max_seq_len,s1_idx=0, s2_idx=None, label_idx=1, skip_rows=1)
    # assert lenght of s1 is more than 1, length of s2 is nothing,
    # label_idx is 1, and then theheaders is not there.
    assert
