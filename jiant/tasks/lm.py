"""Task definitions for language modeling tasks."""
import math
import os
from typing import Iterable, Sequence, Type

# Fields for instance processing
from allennlp.data import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.metrics import Average

from jiant.utils.data_loaders import tokenize_and_truncate
from jiant.tasks.registry import register_task
from jiant.tasks.tasks import (
    UNK_TOK_ALLENNLP,
    UNK_TOK_ATOMIC,
    SequenceGenerationTask,
    atomic_tokenize,
    sentence_to_text_field,
)


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
        self.files_by_split = {
            "train": os.path.join(path, "train.txt"),
            "val": os.path.join(path, "valid.txt"),
            "test": os.path.join(path, "test.txt"),
        }

    def count_examples(self):
        """Computes number of samples
        Assuming every line is one example.
        """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for _ in open(split_path))
        self.example_counts = example_counts

    def get_metrics(self, reset=False):
        """Get metrics specific to the task
        Args:
            reset: (boolean) reset any accumulators or internal state
        """
        nll = self.scorer1.get_metric(reset)
        return {"perplexity": math.exp(nll)}

    def load_data(self):
        # Data is exposed as iterable: no preloading
        pass

    def get_data_iter(self, path):
        """Loading data file and tokenizing the text
        Args:
            path: (str) data file path
        """
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                yield tokenize_and_truncate(self._tokenizer_name, toks, self.max_seq_len)

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """Process a language modeling split by indexing and creating fields.
        Args:
            split: (list) a single list of sentences
            indexers: (Indexer object) indexer to index input words
        """

        def _make_instance(sent_):
            """ Forward targs adds <s> as a target for input </s>
            and bwd targs adds </s> as a target for input <s>
            to avoid issues with needing to strip extra tokens
            in the input for each direction """
            sent_ = model_preprocessing_interface.boundary_token_fn(sent_)  # Add <s> and </s>
            d = {
                "input": sentence_to_text_field(sent_, indexers),
                "targs": sentence_to_text_field(sent_[1:] + [sent_[0]], self.target_indexer),
                "targs_b": sentence_to_text_field([sent_[-1]] + sent_[:-1], self.target_indexer),
            }
            return Instance(d)

        for sent in split:
            yield _make_instance(sent)

    def get_split_text(self, split: str):
        """Get split text as iterable of records.
        Args:
            split: (str) should be one of 'train', 'val', or 'test'.
        """
        return self.get_data_iter(self.files_by_split[split])

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """Yield sentences, used to compute vocabulary.
        """
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent in self.get_data_iter(path):
                yield sent


# TODO: restructure LM task hierarchy
@register_task("bwb", rel_path="BWB/")
class WikiTextLMTask(LanguageModelingTask):
    """ Language modeling on a Wikitext dataset
    See base class: LanguageModelingTask
    """

    def get_data_iter(self, path):
        """ Rather than return a whole list of examples, stream them """
        nonatomics_toks = [UNK_TOK_ALLENNLP, "<unk>"]
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                # WikiText103 preprocesses unknowns as '<unk>'
                # which gets tokenized as '@', '@', 'UNKNOWN', ...
                # We replace to avoid that
                sent = atomic_tokenize(
                    toks,
                    UNK_TOK_ATOMIC,
                    nonatomics_toks,
                    self.max_seq_len,
                    tokenizer_name=self._tokenizer_name,
                )
                # we also filtering out headers (artifact of the data)
                # which are processed to have multiple = signs
                if sent.count("=") >= 2 or len(toks) < self.min_seq_len + 2:
                    continue
                yield sent


@register_task("wiki103", rel_path="WikiText103/")
class WikiText103LMTask(WikiTextLMTask):
    """Language modeling task on Wikitext 103
    See base class: WikiTextLMTask
    """

    def __init__(self, path, *args, **kw):
        super().__init__(path, *args, **kw)
        self.files_by_split = {
            "train": os.path.join(path, "train.sentences.txt"),
            "val": os.path.join(path, "valid.sentences.txt"),
            "test": os.path.join(path, "test.sentences.txt"),
        }


class MaskedLanguageModelingTask(LanguageModelingTask):
    """
       Generic Masked Language Modeling Task
    """

    pass


@register_task("mlm", rel_path="WikiText103/")
class MLMTask(MaskedLanguageModelingTask):
    """
    Masked language modeling task on Toronto Books dataset
    Attributes:
        max_seq_len: (int) maximum sequence length
        min_seq_len: (int) minimum sequence length
        files_by_split: (dict) files for three data split (train, val, test)
    """

    def __init__(self, path, *args, **kw):
        super().__init__(path, *args, **kw)
        self.files_by_split = {
            "train": os.path.join(path, "train.sentences.txt"),
            "val": os.path.join(path, "valid.sentences.txt"),
            "test": os.path.join(path, "test.sentences.txt"),
        }

    def update_metrics(self, out, batch=None):
        #self.scorer1(logits,labels)
        self.scorer1(out["loss"].mean())
        return

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """Process a language modeling split by indexing and creating fields.
        Args:
            split: (list) a single list of sentences
            indexers: (Indexer object) indexer to index input words
        """

        def _make_instance(sent_):
            """ Forward targs adds <s> as a target for input </s>
            and bwd targs adds </s> as a target for input <s>
            to avoid issues with needing to strip extra tokens
            in the input for each direction """
            sent_ = model_preprocessing_interface.boundary_token_fn(sent_)  # Add <s> and </s>
            d = {
                "input": sentence_to_text_field(sent_, indexers),
                "targs": sentence_to_text_field(sent_, indexers),
            }
            return Instance(d)

        for sent in split:
            yield _make_instance(sent)



@register_task("mlm_toronto", rel_path="toronto/")
class TorontoLanguageModelling(MaskedLanguageModelingTask):
    """ Language modeling on the Toronto Books dataset
    See base class: LanguageModelingTask
    """

    def get_data_iter(self, path):
        """Load data file, tokenize text and concat sentences to create long term dependencies.
        Args:
            path: (str) data file path
        """
        seq_len = self.max_seq_len
        tokens = []
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                toks_v = toks.split()
                toks = toks.split() + ["<EOS>"]
                tokens += toks
            for i in range(0, len(tokens), seq_len):
                yield tokens[i : i + seq_len]
