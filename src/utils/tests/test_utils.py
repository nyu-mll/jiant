import utils
import unittest
from io import StringIO

class TestLoadTsvLablesOneSentence(unittest.TestCase):
    def test(self):
        mock_csv = StringIO("sentence	label",
        "it 's a charming and often affecting journey . 	1",
        "unflinchingly bleak and desperate 	0")
        with open('temp_output.tsv', 'w', newline='') as f_output:
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(mock_csv)
        max_seq_len = 30
        import pdb; pdb.set_trace()
        sent1s, sent2s, labels = utils.load_tsv('data_output.tsv', max_seq_len,s1_idx=0, s2_idx=None, label_idx=1, skip_rows=1)
        assert len(sent1s) == 2, "This sentence is supposed to match"
        assert len(sent2s) == 0, "This sentence is supposed to be nothing"
        assert len(labels) == 2, "And the labels are great."
