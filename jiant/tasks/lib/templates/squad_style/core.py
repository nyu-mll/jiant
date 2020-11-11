import json
import numpy as np

import torch
from dataclasses import dataclass
from typing import Union, List, Dict, Optional

from transformers.tokenization_bert import whitespace_tokenize

from jiant.tasks.lib.templates.squad_style import utils as squad_utils
from jiant.shared.constants import PHASE
from jiant.tasks.core import (
    BaseExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.utils.python.datastructures import ExtendedDataClassMixin
from jiant.utils.display import maybe_tqdm

import logging

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart"}

logger = logging.getLogger(__name__)


@dataclass
class Example(BaseExample):
    # For training examples, we usually have `answer_text` and `start_position_character`
    # For eval examples, we usually has answers, a list of dicts with keys
    #   ["answer_start", "text"]
    qas_id: str
    question_text: str
    context_text: str
    answer_text: Optional[str]
    start_position_character: Optional[int]
    title: str
    answers: Optional[list]
    is_impossible: bool

    # ===
    doc_tokens: Optional[list] = None
    char_to_word_offset: Optional[list] = None
    start_position: int = 0
    end_position: int = 0

    def __post_init__(self):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if self.start_position_character is not None and not self.is_impossible:
            self.start_position = char_to_word_offset[self.start_position_character]
            self.end_position = char_to_word_offset[
                min(
                    self.start_position_character + len(self.answer_text) - 1,
                    len(char_to_word_offset) - 1,
                )
            ]

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
            cleaned_answer_text = " ".join(whitespace_tokenize(self.answer_text))
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

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, self.answer_text
            )

        spans = []

        truncated_query = tokenizer.encode(
            self.question_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_query_length,
        )

        # Tokenizers who insert 2 SEP tokens in-between <context> & <question>
        #   need to have special handling
        # in the way they compute mask of added tokens.
        tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
        sequence_added_tokens = (
            tokenizer.max_len - tokenizer.max_len_single_sentence + 1
            if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
            else tokenizer.max_len - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
                truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
                span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
                truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
                pad_to_max_length=True,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                stride=max_seq_length
                - doc_stride
                - len(truncated_query)
                - sequence_pair_added_tokens,
                return_token_type_ids=True,
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
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
                index = (
                    len(truncated_query) + sequence_added_tokens + i
                    if tokenizer.padding_side == "right"
                    else i
                )
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = (
                len(truncated_query) + sequence_added_tokens
            )
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
                is_max_context = _new_check_is_max_context(
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
                p_mask[len(truncated_query) + sequence_added_tokens :] = 0
            else:
                p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

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
                        doc_offset = len(truncated_query) + sequence_added_tokens

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
class DataRow(BaseDataRow):
    unique_id: str
    qas_id: str
    tokens: list
    token_to_orig_map: dict
    token_is_max_context: dict
    input_ids: np.array
    input_mask: np.array
    segment_ids: np.array
    cls_index: np.array
    p_mask: np.array
    paragraph_len: int
    start_position: int
    end_position: int
    answers: list
    doc_tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    start_position: torch.LongTensor
    end_position: torch.LongTensor
    cls_index: torch.LongTensor
    p_mask: torch.FloatTensor
    tokens: list


class BaseSquadStyleTask(Task):
    Example = NotImplemented
    DataRow = NotImplemented
    Batch = NotImplemented

    TASK_TYPE = TaskTypes.SQUAD_STYLE_QA

    def __init__(
        self,
        name,
        path_dict,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        doc_stride=128,
        max_query_length=64,
    ):
        """SQuAD-style Task object, with support for both SQuAD v1.1 and SQuAD v2.0 formats

        Args:
            name (str): task_name
            path_dict (Dict[str, str]): Dictionary to paths to data
            version_2_with_negative (bool): Whether negative (impossible-to-answer) is an option.
                                            False for SQuAD v1.1-type tasks
                                            True for SquAD 2.0-type tasks
            n_best_size (int): The total number of n-best predictions to generate in the
                               n-best predictions.
            max_answer_length (int): The maximum length of an answer that can be generated.
                                     This is needed because the start and end predictions are
                                     not conditioned on one another.
            null_score_diff_threshold (float): If null_score - best_non_null is greater than
                                               the threshold predict null.
            doc_stride (int): When splitting up a long document into chunks, how much stride
                              to take between chunks.
            max_query_length (int): The maximum number of tokens for the question. Questions
                                    longer than this will be truncated to this length.
        """
        super().__init__(name=name, path_dict=path_dict)
        self.version_2_with_negative = version_2_with_negative
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.null_score_diff_threshold = null_score_diff_threshold

        # Tokenization hyperparameters
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length

    def get_train_examples(self):
        return self.read_squad_examples(path=self.train_path, set_type=PHASE.TRAIN)

    def get_val_examples(self):
        return self.read_squad_examples(path=self.val_path, set_type=PHASE.VAL)

    def get_test_examples(self):
        return self.read_squad_examples(path=self.test_path, set_type=PHASE.TEST)

    @classmethod
    def read_squad_examples(cls, path, set_type):
        return generic_read_squad_examples(path=path, set_type=set_type, example_class=cls.Example,)


def generic_read_squad_examples(
    path: str, set_type: str, example_class: type = dict, read_title: bool = True
):

    with open(path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]

    is_training = set_type == PHASE.TRAIN
    examples = []
    for entry in maybe_tqdm(input_data, desc="Reading SQuAD Entries"):
        if read_title:
            title = entry["title"]
        else:
            title = "-"
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                if "is_impossible" in qa:
                    is_impossible = qa["is_impossible"]
                else:
                    is_impossible = False

                if not is_impossible:
                    if is_training:
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                    else:
                        answers = qa["answers"]

                example = example_class(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )
                examples.append(example)
    return examples


@dataclass
class PartialDataRow(ExtendedDataClassMixin):
    qas_id: str
    doc_tokens: List[str]
    tokens: List[str]
    token_to_orig_map: Dict[int, int]
    token_is_max_context: Dict[int, bool]
    doc_tokens: List[str]
    answers: List[Dict]

    @classmethod
    def from_data_row(cls, data_row: DataRow):
        data_row_dict = data_row.to_dict()
        return PartialDataRow(**{k: data_row_dict[k] for k in cls.get_fields()})


def data_rows_to_partial_examples(
    data_rows: List[PartialDataRow],
) -> List[squad_utils.PartialExample]:
    qas_id_to_data_rows = {}
    for i, data_row in enumerate(data_rows):
        data_row.unique_id = 1000000000 + i
        if data_row.qas_id not in qas_id_to_data_rows:
            qas_id_to_data_rows[data_row.qas_id] = []
        qas_id_to_data_rows[data_row.qas_id].append(data_row)
    partial_examples = []
    for qas_id in sorted(list(qas_id_to_data_rows.keys())):
        first_data_row = qas_id_to_data_rows[qas_id][0]
        partial_examples.append(
            squad_utils.PartialExample(
                doc_tokens=first_data_row.doc_tokens,
                qas_id=first_data_row.qas_id,
                partial_features=[
                    squad_utils.PartialFeatures(
                        unique_id=data_row.unique_id,
                        tokens=data_row.tokens,
                        token_to_orig_map=data_row.token_to_orig_map,
                        token_is_max_context=data_row.token_is_max_context,
                    )
                    for data_row in qas_id_to_data_rows[qas_id]
                ],
                answers=first_data_row.answers,
            )
        )
    return partial_examples


def is_whitespace(c_):
    if c_ == " " or c_ == "\t" or c_ == "\r" or c_ == "\n" or ord(c_) == 0x202F:
        return True
    return False


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def logits_to_pred_results_list(logits):
    """Convert logits to preds
    :param logits: np.ndarray (batch_size, 2, seq_len)
    :return: List[squad_utils.SquadResult]
    """
    return [
        squad_utils.SquadResult(
            unique_id=1000000000 + i, start_logits=logits[i, 0], end_logits=logits[i, 1],
        )
        for i in range(logits.shape[0])
    ]


def compute_predictions_logits_v3(
    data_rows: List[Union[PartialDataRow, DataRow]],
    logits: np.ndarray,
    n_best_size,
    max_answer_length,
    do_lower_case,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
    skip_get_final_text=False,
    verbose=True,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    partial_examples = data_rows_to_partial_examples(data_rows)
    all_pred_results = logits_to_pred_results_list(logits)
    predictions = squad_utils.compute_predictions_logits_v2(
        partial_examples=partial_examples,
        all_results=all_pred_results,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        do_lower_case=do_lower_case,
        version_2_with_negative=version_2_with_negative,
        null_score_diff_threshold=null_score_diff_threshold,
        tokenizer=tokenizer,
        verbose=verbose,
        skip_get_final_text=skip_get_final_text,
    )
    results = squad_utils.squad_evaluate(partial_examples, predictions)
    return results, predictions
