import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from jiant.shared.constants import PHASE
from jiant.tasks.lib.templates.squad_style import core as squad_style_template
from jiant.utils.python.io import read_json
from jiant.utils.display import maybe_tqdm
from jiant.utils.python.datastructures import take_one

logger = logging.getLogger(__name__)


@dataclass
class Example(squad_style_template.Example):
    # Additional fields
    background_text: Optional[str] = None
    situation_text: Optional[str] = None

    def tokenize(self, tokenizer):
        raise NotImplementedError("SQuaD is weird")

    def to_feature_list(
        self, tokenizer, max_seq_length, doc_stride, max_query_length, set_type,
    ):
        is_training = set_type == PHASE.TRAIN
        features = []
        if is_training and not self.is_impossible:
            # Get start and end position
            start_position = self.start_position
            end_position = self.end_position

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(self.doc_tokens[start_position : (end_position + 1)])
            cleaned_answer_text = " ".join(
                squad_style_template.whitespace_tokenize(self.answer_text)
            )
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning(
                    "Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text
                )
                return []

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(self.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not self.is_impossible:
            tok_start_position = orig_to_tok_index[self.start_position]
            if self.end_position < len(self.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[self.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            # noinspection PyProtectedMember
            (tok_start_position, tok_end_position) = squad_style_template._improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, self.answer_text
            )

        spans = []

        # Usually, SQuAD needs to adjust for document offset with the number of added tokens,
        #   before the context.
        # For BERT-likes, that's
        #    [cls] question [sep] context [sep]
        #    so that's 2 extra tokens before the context.
        # For RoBERTa-likes, that's
        #    <s> question </s> </s> context </s> </s>
        #    so that's 3.
        # In our case however, we have no question, so we have
        #    [cls] context ...
        # so sequence_added_tokens always = 1.
        # (This may not apply for future added models that don't start with a CLS token,
        #   such as XLNet/GPT-2)
        sequence_added_tokens = 1
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
                span_doc_tokens,
                truncation="only_first",
                pad_to_max_length=True,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                stride=max_seq_length - doc_stride - sequence_pair_added_tokens,
                return_token_type_ids=True,
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - sequence_pair_added_tokens,
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                if tokenizer.padding_side == "right":
                    non_padded_ids = encoded_dict["input_ids"][
                        : encoded_dict["input_ids"].index(tokenizer.pad_token_id)
                    ]
                else:
                    last_padding_id_position = (
                        len(encoded_dict["input_ids"])
                        - 1
                        - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                    )
                    non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = 0
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict
                and len(encoded_dict["overflowing_tokens"]) == 0
            ):
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                # noinspection PyProtectedMember
                is_max_context = squad_style_template._new_check_is_max_context(
                    spans, doc_span_index, doc_span_index * doc_stride + j
                )
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer
            #         (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = np.ones_like(span["token_type_ids"])
            if tokenizer.padding_side == "right":
                p_mask[sequence_added_tokens:] = 0
            else:
                p_mask[-len(span["tokens"]) : -sequence_added_tokens] = 0

            pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
            special_token_indices = np.asarray(
                tokenizer.get_special_tokens_mask(
                    span["input_ids"], already_has_special_tokens=True
                )
            ).nonzero()

            p_mask[pad_token_indices] = 1
            p_mask[special_token_indices] = 1

            # Set the cls index to 0: the CLS index can be used for impossible answers
            p_mask[cls_index] = 0

            span_is_impossible = self.is_impossible
            start_position = 0
            end_position = 0
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                # noinspection PyUnboundLocalVariable
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index

                    # We store "is_impossible" at an example level instead
                    # noinspection PyUnusedLocal
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            features.append(
                DataRow(
                    unique_id="",
                    qas_id=self.qas_id,
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    token_is_max_context=span["token_is_max_context"],
                    input_ids=np.array(span["input_ids"]),
                    input_mask=np.array(span["attention_mask"]),
                    segment_ids=np.array(span["token_type_ids"]),
                    cls_index=np.array(cls_index),
                    p_mask=np.array(p_mask.tolist()),
                    paragraph_len=span["paragraph_len"],
                    start_position=start_position,
                    end_position=end_position,
                    answers=self.answers,
                    doc_tokens=self.doc_tokens,
                )
            )
        return features


@dataclass
class DataRow(squad_style_template.DataRow):
    pass


@dataclass
class Batch(squad_style_template.Batch):
    pass


class RopesTask(squad_style_template.BaseSquadStyleTask):
    Example = Example
    DataRow = DataRow
    Batch = Batch

    def __init__(
        self,
        name,
        path_dict,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        include_background=True,
    ):
        super().__init__(
            name=name,
            path_dict=path_dict,
            version_2_with_negative=version_2_with_negative,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            null_score_diff_threshold=null_score_diff_threshold,
        )
        self.include_background = include_background

    def get_train_examples(self):
        return self.read_examples(path=self.train_path, set_type=PHASE.TRAIN)

    def get_val_examples(self):
        return self.read_examples(path=self.val_path, set_type=PHASE.VAL)

    def get_test_examples(self):
        return self.read_examples(path=self.test_path, set_type=PHASE.TEST)

    def read_examples(self, path, set_type):
        input_data = read_json(path, encoding="utf-8")["data"]

        is_training = set_type == PHASE.TRAIN
        examples = []
        data = take_one(input_data)
        for paragraph in maybe_tqdm(data["paragraphs"]):
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                # Because answers can also come from questions, we're going to abuse notation
                #   slightly and put the entire background+situation+question into the "context"
                #   and leave nothing for the "question"
                question_text = " "
                if self.include_background:
                    context_segments = [
                        paragraph["background"],
                        paragraph["situation"],
                        qa["question"],
                    ]
                else:
                    context_segments = [paragraph["situation"], qa["question"]]
                full_context = " ".join(segment.strip() for segment in context_segments)

                if is_training:
                    answer = qa["answers"][0]
                    start_position_character = full_context.find(answer["text"])
                    answer_text = answer["text"]
                    answers = []
                else:
                    start_position_character = None
                    answer_text = None
                    answers = [
                        {"text": answer["text"], "answer_start": full_context.find(answer["text"])}
                        for answer in qa["answers"]
                    ]

                example = Example(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=full_context,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title="",
                    is_impossible=False,
                    answers=answers,
                    background_text=paragraph["background"],
                    situation_text=paragraph["situation"],
                )
                examples.append(example)
        return examples
