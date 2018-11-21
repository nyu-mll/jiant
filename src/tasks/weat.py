''' Task class implementations for WEAT and similar tasks for measuring bias in sentence encoders '''
import os
import logging as log

from allennlp.data import Instance
from allennlp.data.fields import LabelField, MetadataField

from ..utils.utils import assert_for_log, SOS_TOK, EOS_TOK

from .tasks import Task, sentence_to_text_field
from .registry import register_task, REGISTRY

@register_task('weat1', rel_path='WEAT/', version='weat1.txt')
@register_task('weat2', rel_path='WEAT/', version='weat2.txt')
@register_task('weat3', rel_path='WEAT/', version='weat3.txt')
@register_task('weat4', rel_path='WEAT/', version='weat4.txt')
class WEATTask(Task):
    ''' '''
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

        def _make_instance(inp, category, idx):
            d = {}
            d["input"] = sentence_to_text_field(inp, indexers)
            d["sent_str"] = MetadataField(" ".join(inp[1:-1]))
            d["category"] = MetadataField(category)
            d["idx"] = LabelField(idx, label_namespace="idxs",
                                  skip_indexing=True)
            return Instance(d)

        return map(_make_instance, *split)

@register_task('weat1-openai', rel_path='WEAT/', version='weat1.txt.retokenized.OpenAI.BPE')
@register_task('weat2-openai', rel_path='WEAT/', version='weat2.txt.retokenized.OpenAI.BPE')
@register_task('weat3-openai', rel_path='WEAT/', version='weat3.txt.retokenized.OpenAI.BPE')
@register_task('weat4-openai', rel_path='WEAT/', version='weat4.txt.retokenized.OpenAI.BPE')
class OpenAIWEATTask(WEATTask):
    ''' Version of WEAT for BPE-tokenized data. '''
    @property
    def tokenizer_name(self):
        return "OpenAI.BPE"

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
                sents += [[SOS_TOK] + w.split() + [EOS_TOK] for w in words]
                assert len(categories) == len(sents)
        self.test_data_text = [sents, categories, range(len(sents))]
        self.train_data_text = [[], [], []]
        self.val_data_text = [[], [], []]
        log.info("\tFinished loading WEAT data.")

