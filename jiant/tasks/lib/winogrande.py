from dataclasses import dataclass

import numpy as np
from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines

from jiant.tasks.lib.templates.shared import (
    create_input_set_from_tokens_and_segments,
    add_cls_token,
)
from jiant.tasks.utils import truncate_sequences


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return WinograndeTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    def featurize(self, tokenizer, feat_spec):

        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 4  # CLS, SEP-SEP, SEP
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 3  # CLS, SEP, SEP

        input_set_ls = []
        unpadded_inputs_ls = []
        for choice in self.choice_list:
            prompt, choice = truncate_sequences(
                tokens_ls=[self.prompt, choice],
                max_length=feat_spec.max_seq_length - special_tokens_count,
            )
            unpadded_inputs = add_cls_token(
                unpadded_tokens=(
                    # choice
                    choice
                    + [tokenizer.sep_token]
                    + maybe_extra_sep
                    # prompt
                    + prompt
                    + [tokenizer.sep_token]
                ),
                unpadded_segment_ids=(
                    # choice
                    [feat_spec.sequence_a_segment_id] * (len(choice) + 1)
                    + maybe_extra_sep_segment_id
                    # prompt + sep
                    + [feat_spec.sequence_b_segment_id] * (len(prompt) + 1)
                ),
                tokenizer=tokenizer,
                feat_spec=feat_spec,
            )
            input_set = create_input_set_from_tokens_and_segments(
                unpadded_tokens=unpadded_inputs.unpadded_tokens,
                unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
                tokenizer=tokenizer,
                feat_spec=feat_spec,
            )
            input_set_ls.append(input_set)
            unpadded_inputs_ls.append(unpadded_inputs)

        return DataRow(
            guid=self.guid,
            input_ids=np.stack([input_set.input_ids for input_set in input_set_ls]),
            input_mask=np.stack([input_set.input_mask for input_set in input_set_ls]),
            segment_ids=np.stack([input_set.segment_ids for input_set in input_set_ls]),
            label_id=self.label_id,
            tokens_list=[unpadded_inputs.unpadded_tokens for unpadded_inputs in unpadded_inputs_ls],
        )


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class WinograndeTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = [1, 2]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []

        for i, line in enumerate(lines):
            choice_pre, option = line["sentence"].split("_")

            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    prompt=option,
                    choice_list=[choice_pre + line["option1"], choice_pre + line["option2"]],
                    label=int(line["answer"]) if set_type != "test" else cls.CHOICE_KEYS[-1],
                )
            )
        return examples
