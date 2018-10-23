import json
import sys
sys.path.append("..")
import utils

def get_dpr_text():
    text2examples = {}
    curr = {}
    for line in open("recast1/dpr_data.txt"):
        line = line.strip()
        if not line:
            # store curr
            curr_text = curr["text"]
            if curr_text not in text2examples:
                text2examples[curr_text] = []
            text2examples[curr_text].append(curr)
            # make new curr
            curr = {}
        else:
            #get id
            line = line.split(":")
            key = line[0].strip()
            val = " ".join(line[1:]).strip()
            curr[key] = val
    return text2examples

def convert_text_examples_to_json(text, example):
   #dict_ke`ys(['provenance', 'index', 'text', 'hypothesis', 'entailed', 'partof'])
   #This assert makes sure that no text appears in train and test
    split = set([ex['partof'] for ex in example])
    assert len(split) == 1
    obj =  {"text": text,
            "info": {'split': list(split)[0],
                    'source': 'recast-dpr'},
            "targets": []
          }
    text = text.split()
    for ex in example:
        hyp = utils.TOKENIZER.tokenize(ex['hypothesis'])
        #obj['targets'].append(ex['hypothesis'])
        assert len(text) <= len(hyp)
        found_diff_word = False
        for idx, pair in enumerate(zip(text, hyp)):
            if pair[0] != pair[1]:
                referent = ''
                found_diff_word = True
                distance = len(hyp) - len(text) + 1
                pro_noun = text[idx]
                found_referent = False
                for word_idx in range(idx+1):
                    referent = hyp[idx:idx+distance]
                    if word_idx == 0:
                        referent[0] = referent[0][0].upper() + referent[0][1:]
                    if referent == text[word_idx:word_idx + distance]:
                        found_referent = True
                        target = {'span1': [idx,idx+1],
                                  'span2': [word_idx,word_idx + distance],
                                  'label': ex['entailed'],
                                  'span1_text': pro_noun,
                                  'span2_text': " ".join(text[word_idx:word_idx + distance])
                                 }
                        obj['targets'].append(target)
                        break;
                #if not found_referent:
                #    import pdb; pdb.set_trace()
                break;

    return obj

def main():
    outfiles = {}
    text2examples = get_dpr_text()
    count = 0
    with open('dpr_probing.json', 'w') as outfile:
        for text, example in text2examples.items():
            text = " ".join(utils.TOKENIZER.tokenize(text))
            obj = convert_text_examples_to_json(text, example)
            if obj["info"]["split"] not in outfiles:
            	outfiles[obj["info"]["split"]] = open('dpr_probing_%s.json' % (obj["info"]["split"]), 'w')
            if not obj['targets']:
                count += 1
                continue
            json.dump(obj, outfiles[obj["info"]["split"]])
            outfiles[obj["info"]["split"]].write('\n')
    print(count)


if __name__ == '__main__':
    main()
