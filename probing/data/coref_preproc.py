
# usage: replace the next couple of lines. Then run:
# `python coref_preproc.py OUTPUT_FILE_NAME ID_FILE.id`
DATA_PREFIX = "/export/corpora/LDC/LDC2013T19/ontonotes-release-5.0/data/files/"
DATA_SUFFIX = ".onf"

import glob
import json
import re
import sys
import numpy as np

counter = 0
lengths = []

def force_tuple(pair):
  indices = pair.split('-')
  return (force_int(indices[0]),
          force_int(indices[1]))

def reindex_spans(spans, offsets, sentence):
  output = []
  for p, (t1, t2) in spans:
    output.append((p, (offsets[0][t1], offsets[1][t2])))
  return output

def process_leaves(iterator):
  sentence = []
  spans = []
  left_offsets = {}
  right_offsets = {}
  offset_key = 0
  for line in iterator:
    if line.strip() == "":
      return " ".join(sentence), reindex_spans(spans,
                                               (left_offsets, right_offsets),
                                               sentence)
    # regex solution doesn't even work well
    # coref_match = re.match(r" *coref:[a-zA-Z ]+([0-9 \-]+).*", line)
    # if coref_match:
    #  items = coref_match.groups(0)[0].split()
    if "coref:" in line:
      items = line[31:].split()[:2]
      spans.append((items[0], force_tuple(items[1])))
    else:
      a = line.strip().split()
      if len(a) == 2:
        try:
          orig = int(a[0])
        except:
          continue
        if not (a[1] == "0" or
                a[1].startswith("*EXP*") or
                a[1].startswith("*PRO*") or
                a[1].startswith("*T*") or
                a[1].startswith("*ICH*") or
                a[1].startswith("*NOT*") or
                a[1].startswith("*-") or
                a[1] == ("*") or
                a[1].startswith("*RNR*") or
                a[1].startswith("*U*")):
          left_offsets[orig] = offset_key
          right_offsets[orig] = offset_key
          sentence.append(a[1])
          offset_key += 1
        else:
          left_offsets[orig] = offset_key
          right_offsets[orig] = offset_key - 1
          continue
  else:  
    assert False, "expected to encounter a newline"

def find_links(span_list):
  pairs = []
  for i, span_1 in enumerate(span_list):
    for span_2 in span_list[i+1:]:
      pairs.append((span_1[1], span_2[1], int(span_1[0] == span_2[0])))
  return pairs
  
def process_file(f):
  all_lines_iter = iter(open(f).readlines())
  sentence_flag = False
  in_coref = False
  annotations_list = []
  for line in all_lines_iter:
    #if line.strip() == "Treebanked sentence:":
    #  next(all_lines_iter)
    #  sentence = next(all_lines_iter).strip()
    if line.strip() == "Leaves:":
      # get list of entity spans
      sentence, span_list = process_leaves(all_lines_iter)
      # do the n^2 thing
      spans = find_links(span_list)
      annotations_list.append((sentence, spans))
  return annotations_list

def force_int(s, p=True):
  s = str(s)
  if s == "":
    raise Exception
  if len(str(s)) >= 3:
    s = s[:3]
  try:
    return int(s)
  except:
    if p:
      print ("coercing {}".format(s))
  return force_int(s[:-1], p=False)

def jsonify(sentence, spans, fname):
  global counter, lengths
  new_entry = {}
  new_entry["text"] = sentence
  lengths.append([len(sentence), len(sentence.split())])
  def correct(span):
    global counter
    counter += 1
    try:
      return {"span1": [force_int(span[0][0]), force_int(span[0][1]) + 1],
              "span2": [force_int(span[1][0]), force_int(span[1][1]) + 1],
              "label": str(span[2])}
    except Exception as e:
      print (sentence, spans, fname, e)
      exit(1)
  new_entry["targets"] = [correct(span) for span in spans]
  new_entry["source"] = fname
  return json.dumps(new_entry)

if __name__ == "__main__":
  output_file = open(sys.argv[1], 'w+')
  id_file = open(sys.argv[2], 'r').read().splitlines()
  for gname in id_file:
    f = DATA_PREFIX + gname + DATA_SUFFIX
    # sentences is a newline separate file; each line contains a json
    # string following the format of edge probing
    annotated_sentences = process_file(f)
    output_file.write("\n".join([jsonify(a, s, gname) for a, s in annotated_sentences]))
    output_file.write("\n")
  print (counter)
  print (np.mean(lengths, axis=0), np.std(lengths, axis=0), np.median(lengths, axis=0))
  print(np.histogram([l[1] for l in lengths]))
