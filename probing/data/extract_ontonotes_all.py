from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
import sys
import numpy as np
import json

TYPE=# fill in with "ner" or "const"

ontonotes = Ontonotes()
file_path = sys.argv[1] # e.g. test/train/development: /nfs/jsalt/home/pitrack/ontonotes/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development
out_file = open(sys.argv[2], 'w+')
ontonotes_reader = ontonotes.dataset_iterator(file_path=sys.argv[1])

counter = []

def jsonify(ners, sentence):
    new_entry = {}
    new_entry["text"] = " ".join(sentence.words)
    def correct(span):
        # spans are of form (label, (begin, end)) (inclusive)
        # and get converted to (right-exclusive)
        #   {"span1": [begin, end], "label": label}
        return {"span1": [span[1][0], span[1][1] + 1],
                "label": span[0]}
    new_entry["targets"] = [correct(span) for span in ners]
    new_entry["source"] = "{} {}".format(sentence.document_id, sentence.sentence_id)
    return new_entry

def get_ners(sentence):
    global counter
    
    spans = bio_tags_to_spans(sentence.named_entities)
    counter.append(len(spans))
    return spans


def nltk_tree_to_spans(nltk_tree):
    # TODO
    # Input: nltk.tree
    # Output: List[(Str, (Int, Int))] of labelled spans
    # where the first element of the tuple is a string
    # and the second is a [begin, end) tuple specifying the span
    pass
    

def get_consts(sentence):
    global counter
    spans = nltk_tree_to_spans(sentence.parse_tree)
    counter.append(len(spans))
    return spans

sent_counter = 0
for sentence in ontonotes_reader:
    sent_counter += 1
    # returns dict of spans, right-exclusive, STRING labeled with
    # named entity label
    if TYPE == "ner":
        spans = get_ners(sentence)
    elif TYPE == "constituents":
        spans = get_consts(sentence)
    out_file.write(json.dumps(jsonify(spans, sentence)))
    out_file.write("\n")

print ("num entities:{}".format(sum(counter)))
print ("some stats mn|std|md: {} {} {}".format(np.mean(counter), np.std(counter), np.median(counter)))
print ("hist: {}".format(np.histogram(counter)))
print ("num sents: {}".format(sent_counter))
