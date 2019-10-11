import gzip
import json
def preprocess_qasrl_datum(datum):
    return {
        "sentence_tokens": datum["sentenceTokens"],
        "entries": [
            {
                "verb": verb_entry["verbInflectedForms"]["stem"],
                "verb_idx": verb_idx,
                "questions": {
                    question: [
                        [
                            {
                                "tokens": datum["sentenceTokens"][span[0] : span[1] + 1],
                                "span": span,
                            }
                            for span in answer_judgment["spans"]
                        ]
                        for answer_judgment in q_data["answerJudgments"]
                        if answer_judgment["isValid"]
                    ]
                    for question, q_data in verb_entry["questionLabels"].items()
                },
            }
            for verb_idx, verb_entry in datum["verbEntries"].items()
        ],
    }

def _load_file():
    path='qasrl-v2/orig/dev.jsonl.gz'
    example_list = []
    with gzip.open(path) as f:
        lines = f.read().splitlines()
        for line in lines:
            datum = preprocess_qasrl_datum(json.loads(line))
            print("datum: ", datum)
            break

def _load_jsonl():
    path='../jiant-data/SQuAD/train-v2.0.json'

    with open(path) as f:
        data = json.load(f)['data']

    examples = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}

    for article in data:
        for paragraph in article['paragraphs']:
            print("paragraph: ", paragraph)
            sys.exit(0)
            examples['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                examples['qids'].append(qa['id'])
                examples['questions'].append(qa['question'])
                examples['qid2cid'].append(len(examples['contexts']) - 1)
                if 'answers' in qa:
                    examples['answers'].append(qa['answers'])
            break
        break
    print("examples: ", examples)

_load_jsonl() 

