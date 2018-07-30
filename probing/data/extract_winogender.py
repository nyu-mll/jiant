import sys
import re
import numpy as np
import json
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

def fix(word):
  if word == 'male':
    return ['he', 'his', 'him']
  if word == 'female':
    return ['she', 'her']
  if word == 'neutral':
    return ['they', 'their', 'them']
  return [word]

def find_span(word, line):
  spans = []
  split_line = line.split()
  for i, w in enumerate(split_line):
    if w.lower() in fix(word):
      j = i
      if i > 0 and (split_line[i-1] == "The" or
                    split_line[i-1] == "the"):
        j = i - 1
      return (j, i+1)
  print ("{} not found in {}".format(word, line.split()))

def process_file(fname):
  lines = [l.strip().split("\t") for l in open(fname).readlines()]
  jsons = []
  for line in lines[1:]:
    links = []
    sent_id, sent = line
    entities = sent_id.split(' .')
    first = find_span(entities[0], sent)
    second = find_span(entities[1], sent)
    third = find_span(entities[3], sent)
    answer = int(entities[2])
    links.append((first, second, 0))
    links.append((first, third, int(answer == 0)))
    links.append((second, third, int(answer == 1)))
    jsons.append(jsonify(sent, links, fname + "-" + ".".join(entities)))
  return jsons
  
if __name__ == "__main__":
  jsons = process_file(sys.argv[1])
  output_file = open(sys.argv[2], 'w+')
  output_file.write("\n".join([j for j in jsons]))
  output_file.write("\n")  
  print (counter)
  print (np.mean(lengths, axis=0), np.std(lengths, axis=0), np.median(lengths, axis=0))
  print(np.histogram([l[1] for l in lengths]))
