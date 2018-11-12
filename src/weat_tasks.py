''' Task class implementations for WEAT and similar tasks for measuring bias in sentence encoders '''
import os
import logging as log
from .tasks import Task, register_task
from .utils import assert_for_log

@register_task('weat1', rel_path='WEAT/', version='weat1.txt')
@register_task('weat2', rel_path='WEAT/', version='weat2.txt')
@register_task('weat3', rel_path='WEAT/', version='weat3.txt')
@register_task('weat4', rel_path='WEAT/', version='weat4.txt')
class WEATTask(Task):
    ''' '''
    def __init__(self, path, version, name="weat"):
        ''' Initialize the task '''
        super(WEATTask, self).__init__(name)
        self.load_data(path, version)
        self.sentences = self.test_data_text

    def load_data(self, path, version):
        ''' Load the data '''
        cat2text = {}
        data_file = os.path.join(path, version)
        with open(data_file, 'r') as data_fh:
            for row in data_fh:
                category, words = row.strip().split(':')
                words = words.split(',')
                cat2text[category] = words
        assert_for_log(len(cat2text) == 4, "Uh oh!")
        self.cat2text = cat2text
        self.test_data_text = [sent for split in cat2text.values() for sent in split]
        self.train_data_text = []
        self.val_data_text = []
        log.info("\tFinished loading WEAT data.")

    def process_split(self, split, indexers):
        ''' '''
        raise NotImplementedError
