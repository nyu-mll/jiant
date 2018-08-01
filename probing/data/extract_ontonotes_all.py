from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
import sys
import numpy as np
import json
from ptb_process import sent_to_dict

TYPE="const" # fill in with "ner" or "const"

ontonotes = Ontonotes()
file_path = sys.argv[1] # e.g. test/train/development/conll-test: /nfs/jsalt/home/pitrack/ontonotes/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development
out_file = open(sys.argv[2], 'w+')
ontonotes_reader = ontonotes.dataset_iterator(file_path=sys.argv[1])

counter = []
num_span_pairs = 0
num_entities = 0
skip_counter = 0

def jsonify(spans, sentence, two_targets=False):
    global num_span_pairs
    new_entry = {}
    new_entry["text"] = " ".join(sentence.words)
    def correct(span):
        global num_span_pairs
        num_span_pairs += 1
        # spans are of form (label, (begin, end)) (inclusive)
        # and get converted to (right-exclusive)
        #   {"span1": [begin, end], "label": label}
        if two_targets:
            return {"span1": [span[1][0], span[1][1] + 1],
                    "span2": [span[2][0], span[2][1] + 1],
                    "label": span[0]}
        else:
            return {"span1": [span[1][0], span[1][1] + 1],
                    "label": span[0]}
    new_entry["targets"] = [correct(span) for span in spans]
    new_entry["source"] = "{} {}".format(sentence.document_id, sentence.sentence_id)
    return new_entry

def get_ners(sentence):
    global counter
    
    spans = bio_tags_to_spans(sentence.named_entities)
    counter.append(len(spans))
    return spans


def nltk_tree_to_spans(nltk_tree):
    # Input: nltk.tree
    # Output: List[(Str, (Int, Int))] of labelled spans
    # where the first element of the tuple is a string
    # and the second is a [begin, end) tuple specifying the span
    span_dict = sent_to_dict(nltk_tree)
    return span_dict
    

def get_consts(sentence):
    global counter, skip_counter
    try:
        span_dict = nltk_tree_to_spans(sentence.parse_tree)
    except Exception as e:
        skip_counter += 1
        return jsonify([], sentence)
    counter.append(len(span_dict['targets']))
    span_dict["source"] = "{} {}".format(sentence.document_id, sentence.sentence_id)
    return span_dict

def find_links(span_list):
  pairs = []
  for i, span_1 in enumerate(span_list):
    for span_2 in span_list[i+1:]:
        pairs.append((str(int(span_1[0] == span_2[0])),
                      span_1[1],
                      span_2[1]))
  return pairs


def get_corefs(sentence):
    global counter
    spans = find_links(list(sentence.coref_spans))
    counter.append(len(spans))
    return spans

sent_counter = 0
for sentence in ontonotes_reader:
    sent_counter += 1
    # returns dict of spans, right-exclusive, STRING labeled with
    # named entity label
    if TYPE == "ner":
        spans = get_ners(sentence)
        out_file.write(json.dumps(jsonify(spans, sentence, two_targets=False)))
    elif TYPE == "const":
        spans_dict = get_consts(sentence)
        out_file.write(json.dumps(spans_dict))
    elif TYPE == "coref":
        spans = get_corefs(sentence)
        num_entities += len(sentence.coref_spans)
        out_file.write(json.dumps(jsonify(spans, sentence, two_targets=True)))
    out_file.write("\n")

print ("num entities:{}".format(sum(counter)))
print ("some stats mn|std|md: {} {} {}".format(np.mean(counter), np.std(counter), np.median(counter)))
print ("hist: {}".format(np.histogram(counter)))
print ("num sents: {}".format(sent_counter))
print ("num_span_pairs: {}".format(num_span_pairs))
print ("also num ents: {}".format(num_entities))
print ("skipped: {}".format(skip_counter))
