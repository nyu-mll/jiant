import sys
import re
import json
import numpy as np
from collections import defaultdict

counter = 0
lengths = []


def jsonify(sentence, spans, fname):
  global counter, lengths
  new_entry = {}
  new_entry["text"] = sentence
  lengths.append([len(sentence), len(sentence.split())])
  def correct(span):
    global counter
    counter += 1
    try:
      return {"span1": [span[0][0], span[0][1]],
              "span2": [span[1][0], span[1][1]],
              "label": str(span[2])}
    except Exception as e:
      print (sentence, spans, fname, e)
      exit(1)
  new_entry["targets"] = [correct(span) for span in spans]
  new_entry["source"] = fname
  return json.dumps(new_entry)

def get_span_idx(line):
  spans = []
  new_sent = []
  for i, word in enumerate(line):
    new_word = word
    if word[0] == "[":
      new_word = new_word[1:]
      spans.append(i)
    if word[-1] == "]":
      new_word = new_word[:-1]
      spans.append(i+1)
    new_sent.append(new_word)
  assert len(spans) % 2 == 0, "{} {}".format(spans, line)
  return ([(spans[2*i], spans[2*i+1]) for i in range(len(spans) / 2)], new_sent)

def process_file(fname):
  lines = [l.split() for l in open(fname).readlines()]
  jsons = []
  for line in lines:
    links = []
    sent_id, sent = line[0], line[1:]
    spans, new_sent = get_span_idx(line[1:])
    try:
      assert len(spans) == 2, "dun goofed {} {}".format(spans, line)
    except:
      print ("dun goofed {} {}".format(spans, line))
      pass
    links.append((spans[0], spans[1], 1))
    if len(spans) == 3:
      links.append((spans[0], spans[2], 1))
      links.append((spans[1], spans[2], 1))
    jsons.append(jsonify(' '.join(new_sent), links, fname + ":" + str(sent_id)))
  return jsons
  
if __name__ == "__main__":
  output_file = open(sys.argv[1], 'w+')
  for fname in sys.argv[2:]:
    annotated_sentences = process_file(fname)
    output_file.write("\n".join(annotated_sentences))
    output_file.write("\n")  
    print (counter)
    print (np.mean(lengths, axis=0), np.std(lengths, axis=0), np.median(lengths, axis=0))
    print(np.histogram([l[1] for l in lengths]))
