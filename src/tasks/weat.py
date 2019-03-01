''' Task class implementations for WEAT and similar tasks for measuring bias in sentence encoders '''
import os
import json
import logging as log

from allennlp.data import Instance
from allennlp.data.fields import LabelField, MetadataField

from ..utils.utils import assert_for_log, SOS_TOK, EOS_TOK

from .tasks import Task, sentence_to_text_field
from .registry import register_task, REGISTRY

@register_task('weat1', rel_path='WEAT/', version='weat1.jsonl')
@register_task('weat2', rel_path='WEAT/', version='weat2.jsonl')
@register_task('weat3', rel_path='WEAT/', version='weat3.jsonl')
@register_task('weat3b', rel_path='WEAT/', version='weat3b.jsonl')
@register_task('weat4', rel_path='WEAT/', version='weat4.jsonl')
@register_task('weat5', rel_path='WEAT/', version='weat5.jsonl')
@register_task('weat5b', rel_path='WEAT/', version='weat5b.jsonl')
@register_task('weat6', rel_path='WEAT/', version='weat6.jsonl')
@register_task('weat6b', rel_path='WEAT/', version='weat6b.jsonl')
@register_task('weat7', rel_path='WEAT/', version='weat7.jsonl')
@register_task('weat7b', rel_path='WEAT/', version='weat7b.jsonl')
@register_task('weat8', rel_path='WEAT/', version='weat8.jsonl')
@register_task('weat8b', rel_path='WEAT/', version='weat8b.jsonl')
@register_task('weat9', rel_path='WEAT/', version='weat9.jsonl')
@register_task('weat10', rel_path='WEAT/', version='weat10.jsonl')
@register_task('sent-weat1', rel_path='WEAT/', version='sent-weat1.jsonl')
@register_task('sent-weat2', rel_path='WEAT/', version='sent-weat2.jsonl')
@register_task('sent-weat3', rel_path='WEAT/', version='sent-weat3.jsonl')
@register_task('sent-weat3b', rel_path='WEAT/', version='sent-weat3b.jsonl')
@register_task('sent-weat4', rel_path='WEAT/', version='sent-weat4.jsonl')
@register_task('sent-weat5', rel_path='WEAT/', version='sent-weat5.jsonl')
@register_task('sent-weat5b', rel_path='WEAT/', version='sent-weat5b.jsonl')
@register_task('sent-weat6', rel_path='WEAT/', version='sent-weat6.jsonl')
@register_task('sent-weat6b', rel_path='WEAT/', version='sent-weat6b.jsonl')
@register_task('sent-weat7', rel_path='WEAT/', version='sent-weat7.jsonl')
@register_task('sent-weat7b', rel_path='WEAT/', version='sent-weat7b.jsonl')
@register_task('sent-weat8', rel_path='WEAT/', version='sent-weat8.jsonl')
@register_task('sent-weat8b', rel_path='WEAT/', version='sent-weat8b.jsonl')
@register_task('sent-weat9', rel_path='WEAT/', version='sent-weat9.jsonl')
@register_task('sent-weat10', rel_path='WEAT/', version='sent-weat10.jsonl')
@register_task('angry_black_woman_stereotype', rel_path='WEAT/', version='angry_black_woman_stereotype.jsonl')
@register_task('angry_black_woman_stereotype_b', rel_path='WEAT/', version='angry_black_woman_stereotype_b.jsonl')
@register_task('sent-angry_black_woman_stereotype', rel_path='WEAT/', version='sent-angry_black_woman_stereotype.jsonl')
@register_task('sent-angry_black_woman_stereotype_b', rel_path='WEAT/', version='sent-angry_black_woman_stereotype_b.jsonl')
@register_task('heilman_double_bind_competent_1+3-', rel_path='WEAT/', version='heilman_double_bind_competent_1+3-.jsonl')
@register_task('heilman_double_bind_competent_1-', rel_path='WEAT/', version='heilman_double_bind_competent_1-.jsonl')
@register_task('heilman_double_bind_competent_1', rel_path='WEAT/', version='heilman_double_bind_competent_1.jsonl')
@register_task('heilman_double_bind_competent_one_sentence', rel_path='WEAT/', version='heilman_double_bind_competent_one_sentence.jsonl')
@register_task('heilman_double_bind_competent_one_word', rel_path='WEAT/', version='heilman_double_bind_competent_one_word.jsonl')
@register_task('sent-heilman_double_bind_competent_one_word', rel_path='WEAT/', version='sent-heilman_double_bind_competent_one_word.jsonl')
@register_task('heilman_double_bind_likable_1+3-', rel_path='WEAT/', version='heilman_double_bind_likable_1+3-.jsonl')
@register_task('heilman_double_bind_likable_1-', rel_path='WEAT/', version='heilman_double_bind_likable_1-.jsonl')
@register_task('heilman_double_bind_likable_1', rel_path='WEAT/', version='heilman_double_bind_likable_1.jsonl')
@register_task('heilman_double_bind_likable_one_sentence', rel_path='WEAT/', version='heilman_double_bind_likable_one_sentence.jsonl')
@register_task('heilman_double_bind_likable_one_word', rel_path='WEAT/', version='heilman_double_bind_likable_one_word.jsonl')
@register_task('sent-heilman_double_bind_likable_one_word', rel_path='WEAT/', version='sent-heilman_double_bind_likable_one_word.jsonl')
class WEATTask(Task):
    ''' Task class for WEAT tests '''
    def __init__(self, path, max_seq_len, version, name="weat"):
        ''' Initialize the task '''
        super(WEATTask, self).__init__(name)
        self.load_data(path, version)
        self.sentences = self.test_data_text[0]

    def load_data(self, path, version):
        ''' Load the data '''
        sents = []
        categories = []
        data_file = os.path.join(path, version)
        with open(data_file, 'r') as data_fh:
            for row in data_fh:
                category, words = row.strip().split(':')
                words = words.split(',')
                categories += [category for _ in range(len(words))]
                sents += [[SOS_TOK] + [w] + [EOS_TOK] for w in words]
                assert len(categories) == len(sents)
        self.test_data_text = [sents, categories, range(len(sents))]
        self.train_data_text = [[], [], []]
        self.val_data_text = [[], [], []]
        log.info("\tFinished loading WEAT data.")

    def process_split(self, split, indexers):
        ''' Process split into iterator of instances '''

        def _make_instance(inp, test_split, category, idx):
            d = {}
            d["input"] = sentence_to_text_field(inp, indexers)
            d["sent_str"] = MetadataField(" ".join(inp[1:-1]))
            d["test_split"] = MetadataField(test_split)
            d["category"] = MetadataField(category)
            d["idx"] = LabelField(idx, label_namespace="idxs",
                                  skip_indexing=True)
            return Instance(d)

        return map(_make_instance, *split)

@register_task('weat1-openai', rel_path='WEAT/', version='weat1.jsonl.retokenized.OpenAI.BPE')
@register_task('weat2-openai', rel_path='WEAT/', version='weat2.jsonl.retokenized.OpenAI.BPE')
@register_task('weat3-openai', rel_path='WEAT/', version='weat3.jsonl.retokenized.OpenAI.BPE')
@register_task('weat4-openai', rel_path='WEAT/', version='weat4.jsonl.retokenized.OpenAI.BPE')
@register_task('weat5-openai', rel_path='WEAT/', version='weat5.jsonl.retokenized.OpenAI.BPE')
@register_task('weat6-openai', rel_path='WEAT/', version='weat6.jsonl.retokenized.OpenAI.BPE')
@register_task('weat7-openai', rel_path='WEAT/', version='weat7.jsonl.retokenized.OpenAI.BPE')
@register_task('weat8-openai', rel_path='WEAT/', version='weat8.jsonl.retokenized.OpenAI.BPE')
@register_task('weat9-openai', rel_path='WEAT/', version='weat9.jsonl.retokenized.OpenAI.BPE')
@register_task('weat10-openai', rel_path='WEAT/', version='weat10.jsonl.retokenized.OpenAI.BPE')
@register_task('weat3b-openai', rel_path='WEAT/', version='weat3b.jsonl.retokenized.OpenAI.BPE')
@register_task('weat5b-openai', rel_path='WEAT/', version='weat5b.jsonl.retokenized.OpenAI.BPE')
@register_task('weat6b-openai', rel_path='WEAT/', version='weat6b.jsonl.retokenized.OpenAI.BPE')
@register_task('weat7b-openai', rel_path='WEAT/', version='weat7b.jsonl.retokenized.OpenAI.BPE')
@register_task('weat8b-openai', rel_path='WEAT/', version='weat8b.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat1-openai', rel_path='WEAT/', version='sent-weat1.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat2-openai', rel_path='WEAT/', version='sent-weat2.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat3-openai', rel_path='WEAT/', version='sent-weat3.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat4-openai', rel_path='WEAT/', version='sent-weat4.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat5-openai', rel_path='WEAT/', version='sent-weat5.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat6-openai', rel_path='WEAT/', version='sent-weat6.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat7-openai', rel_path='WEAT/', version='sent-weat7.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat8-openai', rel_path='WEAT/', version='sent-weat8.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat9-openai', rel_path='WEAT/', version='sent-weat9.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat10-openai', rel_path='WEAT/', version='sent-weat10.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat3b-openai', rel_path='WEAT/', version='sent-weat3b.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat5b-openai', rel_path='WEAT/', version='sent-weat5b.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat6b-openai', rel_path='WEAT/', version='sent-weat6b.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat7b-openai', rel_path='WEAT/', version='sent-weat7b.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-weat8b-openai', rel_path='WEAT/', version='sent-weat8b.jsonl.retokenized.OpenAI.BPE')
@register_task('angry_black_woman_stereotype-openai', rel_path='WEAT/', version='angry_black_woman_stereotype.jsonl.retokenized.OpenAI.BPE')
@register_task('angry_black_woman_stereotype_b-openai', rel_path='WEAT/', version='angry_black_woman_stereotype_b.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-angry_black_woman_stereotype-openai', rel_path='WEAT/', version='sent-angry_black_woman_stereotype.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-angry_black_woman_stereotype_b-openai', rel_path='WEAT/', version='sent-angry_black_woman_stereotype_b.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_competent_1+3--openai', rel_path='WEAT/', version='heilman_double_bind_competent_1+3-.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_competent_1--openai', rel_path='WEAT/', version='heilman_double_bind_competent_1-.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_competent_1-openai', rel_path='WEAT/', version='heilman_double_bind_competent_1.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_competent_one_sentence-openai', rel_path='WEAT/', version='heilman_double_bind_competent_one_sentence.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_competent_one_word-openai', rel_path='WEAT/', version='heilman_double_bind_competent_one_word.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-heilman_double_bind_competent_one_word-openai', rel_path='WEAT/', version='sent-heilman_double_bind_competent_one_word.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_likable_1+3--openai', rel_path='WEAT/', version='heilman_double_bind_likable_1+3-.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_likable_1--openai', rel_path='WEAT/', version='heilman_double_bind_likable_1-.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_likable_1-openai', rel_path='WEAT/', version='heilman_double_bind_likable_1.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_likable_one_sentence-openai', rel_path='WEAT/', version='heilman_double_bind_likable_one_sentence.jsonl.retokenized.OpenAI.BPE')
@register_task('heilman_double_bind_likable_one_word-openai', rel_path='WEAT/', version='heilman_double_bind_likable_one_word.jsonl.retokenized.OpenAI.BPE')
@register_task('sent-heilman_double_bind_likable_one_word-openai', rel_path='WEAT/', version='sent-heilman_double_bind_likable_one_word.jsonl.retokenized.OpenAI.BPE')
class OpenAIWEATTask(WEATTask):
    ''' Version of WEAT for BPE-tokenized data. '''
    @property
    def tokenizer_name(self):
        return "OpenAI.BPE"

    def load_data(self, path, version):
        ''' Load the data '''
        sents = []
        categories = []
        sets = []
        data_file = os.path.join(path, version)
        test_d = json.load(open(data_file))
        for set_name, word_set in test_d.items():
            category = word_set["category"]
            examples = word_set["examples"]
            sents += [[SOS_TOK] + ex.split() + [EOS_TOK] for ex in examples]
            sets += [set_name for _ in range(len(examples))]
            categories += [category for _ in range(len(examples))]
            assert len(categories) == len(sents) == len(sets)
        self.test_data_text = [sents, sets, categories, range(len(sents))]
        self.train_data_text = [[], [], [], []]
        self.val_data_text = [[], [], [], []]
        log.info("\tFinished loading WEAT data.")

