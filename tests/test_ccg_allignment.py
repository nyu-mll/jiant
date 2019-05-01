import unittest
import csv
import jiant.scripts.ccg.align_tags_to_bert as ccg_aligner
from string import ascii_lowercase

TEMP_DATASET_CCG = "temp_ccg_dataset.tsv"


class TestCCGAllignment(unittest.TestCase):
    def setUp(self):
        with open(TEMP_DATASET_CCG, 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(["text", "tags"])
            # test for a text that shouldn't be changed by tokenization
            writer.writerow(["Influential members of the House", "A B C D E "])
            # test for text that should be changed at the end
            writer.writerow(
                ["If more information is needed from you Mr. Ford ", "E F G H I J K L M"])
            # text for text that should be changed at the beginning
            writer.writerow(
                ["Mr. Ford if more information is needed from you", "E F G H I J K L M"])
            writer.writerow(
                ["if more information Mr. Ford is needed from you", "E F G H I J K L M"])

    def test(self):
        max_seq_len = 30
        import pandas as pd
        file = pd.read_csv(TEMP_DATASET_CCG, sep="\t")
        tag_ids = {ascii_lowercase[i].upper(): i for i in range(len((list(ascii_lowercase))))}
        result = ccg_aligner.align_tags_BERT(file, "bert-large-uncased", tag_ids)
        assert result.iloc[0]["text"] == file.iloc[0]["text"]
        assert result.iloc[0]["tags"].split(" ") == ["0", "1", "2", "3", "4"]
        assert result.iloc[1]["tags"].split(
            " ") == ['4', '5', '6', '7', '8', '9', '10', '11', '26', '12']
        assert result.iloc[2]["tags"].split(
            " ") == ['4', '26', '5', '6', '7', '8', '9', '10', '11', '12']
        assert result.iloc[3]["tags"].split(
            " ") == ['4', '5', '6', '7', '26', '8', '9', '10', '11', '12']
