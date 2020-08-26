import numpy as np
import torch
from dataclasses import dataclass
from typing import List
import string

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    SuperGlueMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import (
    labels_to_bimap,
    add_cls_token,
    create_input_set_from_tokens_and_segments,
)
from jiant.tasks.utils import truncate_sequences, ExclusiveSpan
from jiant.utils.python.io import read_json_lines
from jiant.utils import retokenize
from jiant.utils.tokenization_normalization import normalize_tokenizations


@dataclass
class Example(BaseExample):
    guid: str
    sentence1: str
    sentence2: str
    word: str
    span1: ExclusiveSpan
    span2: ExclusiveSpan
    label: str

    def tokenize(self, tokenizer):
        def tokenize_span(tokenizer, sentence: str, char_span: ExclusiveSpan):
            """Tokenizes sentence and projects char_span to token span.

            Args:
                tokenizer (transformers.PreTrainedTokenizer): Tokenizer used
                sentence (str): Sentence to be tokenized
                char_span (ExclusiveSpan): character indexed span for sentence

            Returns:
                sentence_target_tokenization (List[str]): tokenized sentence
                target_span (ExclusiveSpane): token span for sentence
            """
            span_start_idx = len(sentence[: char_span.start].split())
            # If the first word in a span starts with punctuation, the first word will
            # erroneously be split into two strings by .split().
            # ie: 'takeaway' -> ["'", "takeaway"]
            # For span alignment, we start the list index at the punctuation.
            if (span_start_idx != 0) and (sentence[: (char_span.start)][-1] in string.punctuation):
                span_start_idx = span_start_idx - 1
            span_text = sentence[char_span.start : char_span.end]

            sentence_space_tokenization = sentence.split()
            sentence_target_tokenization = tokenizer.tokenize(sentence)
            (
                sentence_normed_space_tokenization,
                sentence_normed_target_tokenization,
            ) = normalize_tokenizations(
                sentence_space_tokenization, sentence_target_tokenization, tokenizer
            )
            span_start_char = len(" ".join(sentence_normed_space_tokenization[:span_start_idx]))
            span_text_char = len(span_text)
            aligner = retokenize.TokenAligner(
                sentence_normed_space_tokenization, sentence_normed_target_tokenization
            )
            target_span = ExclusiveSpan(
                *aligner.project_char_to_token_span(
                    span_start_char, span_start_char + span_text_char
                )
            )
            return sentence_target_tokenization, target_span

        sentence1_target_tokenization, target_span1 = tokenize_span(
            tokenizer, self.sentence1, self.span1
        )
        sentence2_target_tokenization, target_span2 = tokenize_span(
            tokenizer, self.sentence2, self.span2
        )

        return TokenizedExample(
            guid=self.guid,
            sentence1_tokens=sentence1_target_tokenization,
            sentence2_tokens=sentence2_target_tokenization,
            word=tokenizer.tokenize(self.word),  # might be more than one token
            sentence1_span=target_span1,
            sentence2_span=target_span2,
            label_id=WiCTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    sentence1_tokens: List[str]
    sentence2_tokens: List[str]
    word: List[str]
    sentence1_span: ExclusiveSpan
    sentence2_span: ExclusiveSpan
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 6  # CLS, SEP-SEP, SEP-SEP, SEP
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 4  # CLS, SEP, SEP, SEP

        sentence1_tokens, sentence2_tokens = truncate_sequences(
            tokens_ls=[self.sentence1_tokens, self.sentence2_tokens],
            max_length=feat_spec.max_seq_length - len(self.word) - special_tokens_count,
        )

        unpadded_tokens = (
            self.word
            + [tokenizer.sep_token]
            + maybe_extra_sep
            + sentence1_tokens
            + [tokenizer.sep_token]
            + maybe_extra_sep
            + sentence2_tokens
            + [tokenizer.sep_token]
        )
        # Don't have a choice here -- just leave words as part of sent1
        unpadded_segment_ids = (
            [feat_spec.sequence_a_segment_id] * (len(self.word) + 1)
            + maybe_extra_sep_segment_id
            + [feat_spec.sequence_a_segment_id] * (len(sentence1_tokens) + 1)
            + maybe_extra_sep_segment_id
            + [feat_spec.sequence_b_segment_id] * (len(sentence2_tokens) + 1)
        )

        unpadded_inputs = add_cls_token(
            unpadded_tokens=unpadded_tokens,
            unpadded_segment_ids=unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        word_sep_offset = 2 if feat_spec.sep_token_extra else 1
        sent1_sep_offset = 2 if feat_spec.sep_token_extra else 1

        # Both should be inclusive spans at the end
        sentence1_span = ExclusiveSpan(
            start=self.sentence1_span[0]
            + unpadded_inputs.cls_offset
            + word_sep_offset
            + len(self.word),
            end=self.sentence1_span[1]
            + unpadded_inputs.cls_offset
            + word_sep_offset
            + len(self.word),
        ).to_inclusive()
        sentence2_span = ExclusiveSpan(
            start=self.sentence2_span[0]
            + unpadded_inputs.cls_offset
            + word_sep_offset
            + sent1_sep_offset
            + len(self.word)
            + len(sentence1_tokens),
            end=self.sentence2_span[1]
            + unpadded_inputs.cls_offset
            + word_sep_offset
            + sent1_sep_offset
            + len(self.word)
            + len(sentence1_tokens),
        ).to_inclusive()

        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            spans=np.array([sentence1_span, sentence2_span]),
            label_id=self.label_id,
            tokens=unpadded_inputs.unpadded_tokens,
            word=self.word,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.array
    input_mask: np.array
    segment_ids: np.array
    spans: np.array  # num_spans x 2
    label_id: int
    tokens: List
    word: List


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    spans: torch.LongTensor
    label_id: torch.LongTensor
    tokens: List
    word: List


class WiCTask(SuperGlueMixin, Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.SPAN_COMPARISON_CLASSIFICATION
    LABELS = [False, True]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_spans(self):
        return 2

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for line in lines:
            span1 = ExclusiveSpan(int(line["start1"]), int(line["end1"]))
            span2 = ExclusiveSpan(int(line["start2"]), int(line["end2"]))
            # Note, the chosen word may be different (e.g. different tenses) in sent1 and sent2,
            #   hence we don't do an assert here.
            examples.append(
                Example(
                    # NOTE: WiCTask.super_glue_format_preds() is dependent on this guid format.
                    guid="%s-%s" % (set_type, line["idx"]),
                    sentence1=line["sentence1"],
                    sentence2=line["sentence2"],
                    word=line["word"],
                    span1=span1,
                    span2=span2,
                    label=line["label"] if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples

    @classmethod
    def super_glue_format_preds(cls, pred_dict):
        """Reformat this task's raw predictions to have the structure expected by SuperGLUE."""
        lines = []
        for pred, guid in zip(list(pred_dict["preds"]), list(pred_dict["guids"])):
            lines.append({"idx": int(guid.split("-")[1]), "label": str(cls.LABELS[pred]).lower()})
        return lines
