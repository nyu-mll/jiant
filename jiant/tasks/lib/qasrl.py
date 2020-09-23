import gzip
import nltk
import json

from jiant.tasks.lib.templates import span_prediction as span_pred_template
from jiant.utils.retokenize import TokenAligner


class QASRLTask(span_pred_template.AbstractSpanPredictionTask):
    def get_train_examples(self):
        return self._create_examples(self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(self.test_path, set_type="test")

    def _create_examples(self, file_path, set_type):

        with gzip.open(file_path) as f:
            lines = f.read().splitlines()

        examples = []
        ptb_detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()

        for line in lines:
            datum = json.loads(line)
            datum = {
                "sentence_tokens": datum["sentenceTokens"],
                "entries": [
                    {
                        "verb": verb_entry["verbInflectedForms"]["stem"],
                        "verb_idx": verb_idx,
                        "questions": {
                            question: [
                                [
                                    {
                                        "tokens": datum["sentenceTokens"][span[0] : span[1]],
                                        "span": (span[0], span[1] - 1),
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

            passage_ptb_tokens = datum["sentence_tokens"]
            passage_space_tokens = ptb_detokenizer.detokenize(
                passage_ptb_tokens, convert_parentheses=True
            ).split()
            passage_space_str = " ".join(passage_space_tokens)

            token_aligner = TokenAligner(source=passage_ptb_tokens, target=passage_space_tokens)

            for entry in datum["entries"]:
                for question, answer_list in entry["questions"].items():
                    for answer in answer_list:
                        for answer_span in answer:
                            try:
                                answer_char_span = token_aligner.project_token_to_char_span(
                                    answer_span["span"][0], answer_span["span"][1], inclusive=True
                                )
                            except ValueError:
                                continue
                            answer_str = passage_space_str[
                                answer_char_span[0] : answer_char_span[1] + 1
                            ]

                            examples.append(
                                span_pred_template.Example(
                                    guid="%s-%s" % (set_type, len(examples)),
                                    passage=passage_space_str,
                                    question=question,
                                    answer=answer_str,
                                    answer_char_span=answer_char_span,
                                )
                            )

        return examples
