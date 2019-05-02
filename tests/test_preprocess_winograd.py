import csv
import os
import pandas as pd
import shutil
import tempfile
import unittest
import scripts.winograd.preprocess_winograd as preprocess_winograd
import json
import copy 

class TestPreprocessWinograd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_winograd_dataset.tsv")
        with open(self.path, 'w') as jsonfile:

            # test for a indices that shouldn't be changed by tokenization
            jsonfile.write(json.dumps({"text": "Members of the House clapped their hands", 
                        "targets": [{"span1_index":0, "span1_text": "members", 
                                    "span2_index": 5, "span2_text": "their",
                                    "label": True}]}))
            jsonfile.write("\n")
            # test where both span indices should shift
            jsonfile.write(json.dumps({"text": "Mr. Ford told me to tell you to contact him", 
                    "targets": [{"span1_index": 0, "span1_text": "Mr. Ford", 
                                "span2_index": 9, "span2_text": "him",
                                "label": True}]}))
            jsonfile.write("\n")
            # test where only one of the span indices changes
            jsonfile.write(json.dumps({"text": "I told you already, Mr. Ford!", 
                    "targets": [{"span1_index": 4, "span1_text": "Mr. Ford", 
                                "span2_index": 0, "span2_text": "I",
                                "label": False}]}))
            jsonfile.write("\n")


    def test_bert(self):
        records = list(pd.read_json(self.path, lines=True).T.to_dict().values())
        orig_records = copy.deepcopy(records)
        for rec in records:
            preprocess_winograd.realign_spans(rec, "bert-large-cased")
        print(records[0])
        print(orig_records[0])
        assert records[0]["text"] == orig_records[0]["text"]
        # the two below should be changed by tokenization. 
        assert records[1]["text"] != orig_records[1]["text"]
        assert records[2]["text"] != orig_records[2]["text"]

        result_span1 = records[0]["targets"][0]["span1"]
        result_span2 = records[0]["targets"][0]["span2"]
        assert result_span1 == [0, 1]
        assert result_span2 == [5, 6]

        result_span1 = records[1]["targets"][0]["span1"]
        result_span2 = records[1]["targets"][0]["span2"]

        assert result_span1 == [0, 3]
        assert result_span2 == [10, 11]

        result_span1 = records[2]["targets"][0]["span1"]
        result_span2 = records[2]["targets"][0]["span2"]    

        assert result_span1 == [5, 9]
        assert result_span2 == [0, 1]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
