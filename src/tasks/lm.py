"""Task definitions for language modeling tasks."""
import json
import logging as log
import math
import os

from allennlp.training.metrics import Average
from allennlp.data.token_indexers import SingleIdTokenIndexer

# Fields for instance processing
from allennlp.data import Instance, Token

from ..utils.utils import process_sentence, truncate

from typing import Iterable, Sequence, List, Dict, Any, Type

from .tasks import SequenceGenerationTask
from .tasks import sentence_to_text_field, atomic_tokenize
from .tasks import UNK_TOK_ALLENNLP, UNK_TOK_ATOMIC
from .registry import register_task

class LanguageModelingTask(SequenceGenerationTask):
    """Generic language modeling task
    See base class: SequenceGenerationTask
    Attributes:
        max_seq_len: (int) maximum sequence length
        min_seq_len: (int) minimum sequence length
        target_indexer: (Indexer Obejct) Indexer used for target
        files_by_split: (dict) files for three data split (train, val, test)
    """

    def __init__(self, path, max_seq_len, name, **kw):
        """Init class
        Args:
            path: (str) path that the data files are stored
            max_seq_len: (int) maximum length of one sequence
            name: (str) task name
        """
        super().__init__(name, **kw)
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_perplexity" % self.name
        self.val_metric_decreases = True
        self.max_seq_len = max_seq_len
        self.min_seq_len = 0
        self.target_indexer = {"words": SingleIdTokenIndexer(namespace="tokens")}
        self.files_by_split = {'train': os.path.join(path, "train.txt"),
                               'val': os.path.join(path, "valid.txt"),
                               'test': os.path.join(path, "test.txt")}

    def count_examples(self):
        """Computes number of samples
        Assuming every line is one example.
        """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for line in open(split_path))
        self.example_counts = example_counts

    def get_metrics(self, reset=False):
        """Get metrics specific to the task
        Args:
            reset: (boolean) reset any accumulators or internal state
        """
        nll = self.scorer1.get_metric(reset)
        return {'perplexity': math.exp(nll)}

    def load_data(self, path):
        """Loading data file and tokenizing the text
        Args:
            path: (str) data file path
        """
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                yield process_sentence(toks, self.max_seq_len)

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """Process a language modeling split by indexing and creating fields.
        Args:
            split: (list) a single list of sentences
            indexers: (Indexer object) indexer to index input words
        """
        def _make_instance(sent):
            ''' Forward targs adds <s> as a target for input </s>
            and bwd targs adds </s> as a target for input <s>
            to avoid issues with needing to strip extra tokens
            in the input for each direction '''
            d = {}
            d["input"] = sentence_to_text_field(sent, indexers)
            d["targs"] = sentence_to_text_field(sent[1:] + [sent[0]], self.target_indexer)
            d["targs_b"] = sentence_to_text_field([sent[-1]] + sent[:-1], self.target_indexer)
            return Instance(d)
        for sent in split:
            yield _make_instance(sent)

    def get_split_text(self, split: str):
        """Get split text as iterable of records.
        Args:
            split: (str) should be one of 'train', 'val', or 'test'.
        """
        return self.load_data(self.files_by_split[split])

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """Yield sentences, used to compute vocabulary.
        """
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent in self.load_data(path):
                yield sent


# TODO: restructure LM task hierarchy
@register_task('bwb', rel_path='BWB/')
class WikiTextLMTask(LanguageModelingTask):
    """ Language modeling on a Wikitext dataset
    See base class: LanguageModelingTask
    """
    def load_data(self, path):
        ''' Rather than return a whole list of examples, stream them '''
        nonatomics_toks = [UNK_TOK_ALLENNLP, '<unk>']
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                # WikiText103 preprocesses unknowns as '<unk>'
                # which gets tokenized as '@', '@', 'UNKNOWN', ...
                # We replace to avoid that
                sent = atomic_tokenize(toks, UNK_TOK_ATOMIC, nonatomics_toks, self.max_seq_len)
                # we also filtering out headers (artifact of the data)
                # which are processed to have multiple = signs
                if sent.count("=") >= 2 or len(toks) < self.min_seq_len + 2:
                    continue
                yield sent


@register_task('wiki103', rel_path='WikiText103/')
class WikiText103LMTask(WikiTextLMTask):
    """Language modeling task on Wikitext 103
    See base class: WikiTextLMTask
    """

    def __init__(self, path, *args, **kw):
        super().__init__(path, *args, **kw)
        self.files_by_split = {'train': os.path.join(path, "train.sentences.txt"),
                               'val': os.path.join(path, "valid.sentences.txt"),
                               'test': os.path.join(path, "test.sentences.txt")}

@register_task('wsj', rel_path='Penn/')
class WSJLanguageModelling(LanguageModelingTask):
    """ Language modeling on a PTB dataset
    See base class: LanguageModelingTask
    """

    # def __init__(self, path, max_seq_len, name, **kw):
    #     super().__init__(path, max_seq_len, name)

    def load_data(self, path):
        seq_len=self.max_seq_len
        tokens=[]
        with open(path) as txt_fh:
            for row in txt_fh:
                toks=row.strip()
                if not toks:
                    continue
                toks=toks.split()+["<EOS>"]
                tokens+=toks
            num_sent=int(math.ceil(len(tokens)/seq_len))
            for i in range(num_sent):
                yield tokens[i*seq_len:i*seq_len+seq_len]


    def count_examples(self):
        """Computes number of samples
        Assuming every line is one example.
        """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            #example_counts[split] = 
            arr=[line.strip().split()+["<EOS>"] for line in open(split_path)]
            allf=0
            for x in arr:
                allf+=len(x)
            example_counts[split]=int(math.ceil(allf/self.max_seq_len))
        self.example_counts = example_counts

    
    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """Process a language modeling split by indexing and creating fields.
        Args:
            split: (list) a single list of sentences
            indexers: (Indexer object) indexer to index input words
        """
        def _make_instance(sent):
            ''' Forward targs adds <s> as a target for input </s>
            and bwd targs adds </s> as a target for input <s>
            to avoid issues with needing to strip extra tokens
            in the input for each direction '''
            d = {}
            d["input"] = sentence_to_text_field(sent, indexers)
            d["targs"] = sentence_to_text_field(sent, self.target_indexer)
            d["targs_b"] = sentence_to_text_field([sent[-1]] + sent[:-1], self.target_indexer)
            return Instance(d)
        for sent in split:
            yield _make_instance(sent)


