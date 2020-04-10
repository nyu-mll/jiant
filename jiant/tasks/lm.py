"""Task definitions for language modeling tasks."""
import math
import os
import torch
from typing import Iterable, Sequence, Type
import random
<<<<<<< HEAD
=======
import copy
>>>>>>> de3c44a6c2aad2bdfc5bc3047e063fd03c6c4c12

# Fields for instance processing
from allennlp.data import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.metrics import Average
from allennlp.data.fields import SequenceLabelField, LabelField

from jiant.utils.data_loaders import tokenize_and_truncate, get_tokenizer
from jiant.tasks.registry import register_task
from jiant.tasks.tasks import Task
from jiant.tasks.tasks import (
    UNK_TOK_ALLENNLP,
    UNK_TOK_ATOMIC,
    SequenceGenerationTask,
    PairClassificationTask,
    atomic_tokenize,
    sentence_to_text_field,
)
from transformers import XLMRobertaTokenizer


class AutoregressiveLanguageModelingTask(SequenceGenerationTask):
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
        self._label_namespace = self.name + "_labels"
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

    def get_metrics(self, reset=False):
        """Get metrics specific to the task
        Args:
            reset: (boolean) reset any accumulators or internal state
        """
        nll = self.scorer1.get_metric(reset)
        return {"perplexity": math.exp(nll)}

    def load_data(self):
        # Data is exposed as iterable: no preloading
        self.examples_by_split = {}
        for split in self.files_by_split:
            self.examples_by_split[split] = list(self.get_data_iter(self.files_by_split[split]))

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

    def count_examples(self):
        """Computes number of samples
        Assuming every line is one example.
        """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for _ in self.examples_by_split[split])
        self.example_counts = example_counts

    def get_split_text(self, split: str):
        """Get split text as iterable of records.
        Args:
            split: (str) should be one of 'train', 'val', or 'test'.
        """
        return self.examples_by_split[split]

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """Yield sentences, used to compute vocabulary.
        """
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            for sent in self.examples_by_split[split]:
                yield sent


# TODO: restructure LM task hierarchy
@register_task("bwb", rel_path="BWB/")
class WikiTextLMTask(AutoregressiveLanguageModelingTask):
    """ Language modeling on a Wikitext dataset
    See base class: AutoregressiveLanguageModelingTask
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



@register_task("wikipedia_corpus_mlm", rel_path="wikipedia_corpus_small/")
class MaskedLanguageModelingTask(Task):
    """
    Masked language modeling task on Wikipedia dataset
    Attributes:
        max_seq_len: (int) maximum sequence length
        min_seq_len: (int) minimum sequence length
        files_by_split: (dict) files for three data split (train, val, test)
    We are currently using an unpreprocessed version of the Wikipedia corpus
    that consists of 5% of the data. You can generate the data by following the
    instructions from jiant/scripts/mlm.
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
        self._label_namespace = "mlm"
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

    def load_data(self):
        # Data is exposed as iterable: no preloading
        pass

    def get_metrics(self, reset=False):
        """Get metrics specific to the task
        Args:
            reset: (boolean) reset any accumulators or internal state
        """
        nll = self.scorer1.get_metric(reset)
        return {"perplexity": math.exp(nll)}

    def get_all_labels(self):
        """
        For MLM, the label space is the vocabulary space of the input.
        """
        labels = []
        tokenizer = get_tokenizer(self._tokenizer_name)
        vocab_size = len(tokenizer)
        ordered_vocab = tokenizer.convert_ids_to_tokens(range(vocab_size))
        for word in ordered_vocab:
            labels.append(word)
        return labels

    def update_metrics(self, out, batch=None):
        self.scorer1(out["loss"].mean())
        return

    def get_data_iter(self, path):
        """Loading data file and tokenizing the text
        Args:
            path: (str) data file path
        """
        with open(path, "r", encoding="utf-8") as txt_fh:
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
            sent_ = model_preprocessing_interface.boundary_token_fn(sent_)  # Add <s> and </s>

            input_sent = sentence_to_text_field(sent_, indexers)
            d = {
                "input": input_sent,
                "targs": SequenceLabelField(
                    sent_, input_sent, label_namespace=self._label_namespace
                ),
            }
            return Instance(d)

        for sent in split:
            yield _make_instance(sent)

    def count_examples(self):
        """Computes number of samples
        Assuming every line is one example.
        """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for _ in self.get_data_iter(split_path))
        self.example_counts = example_counts

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
            for sent in self.get_data_iter(self.files_by_split[split]):
                yield sent

    def mlm_dynamic_masking(self, inputs, labels, mask_idx, tokenizer_name, sent_encoder):
        """
        This function does dynamic masking as per the RoBERTa paper. Please refer to https://arxiv.org/abs/1907.11692
        for more details.
        Parameters
        ----------
        inputs: torch.Tensor(type=long),
        labels torch.Tensor(type=long),
        mask_idx: int
        tokenizer_name: str,
        sent_encoder: SentenceEncoder
        Returns
        -------
        inputs: input after dynamic masking,
        labels: labels after masking with -100,
        indices_replaced: (testing purposes) indices that will be replaced by mask_idx,
        indices_random: (testing purposes) indices that will be replaced by a random word,
        masked_indices: (testing purposes) indices that the model will have to predict,
        labels_after_shift: (testing purposes) labels after shifting but before masking
        """
        mlm_probability = 0.15
        # We add 2 because we shift the inputs back by 2 in the forward function in sent encoder.
        mask_idx += 2
        tokenizer = get_tokenizer(tokenizer_name)
        # Masking code from https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py
        probability_matrix = torch.full(labels.shape, mlm_probability, device=inputs.device)
        padding_mask = labels.eq(0)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).to(
            device=inputs.device, dtype=torch.uint8
        )
        tokenizer_name = sent_encoder._text_field_embedder.tokenizer_required
        labels_after_shift, _ = sent_encoder._text_field_embedder.correct_sent_indexing(
            {tokenizer_name: labels}
        )
        # We only compute loss on masked tokens
        # nn.CrossEntropy ignores the indices with value = -100 by default.
        # Therefore, we replace non-masked indices with -100 so that they get ignored
        # in loss computation.
        labels = copy.deepcopy(labels_after_shift)
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        bernoulli_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).to(
            device=inputs.device, dtype=torch.uint8
        )
        indices_replaced = bernoulli_mask & masked_indices
        inputs[indices_replaced] = mask_idx

        # 10% of the time, we replace masked input tokens with random word
        bernoulli_mask = torch.bernoulli(torch.full(labels.shape, 0.5)).to(
            device=inputs.device, dtype=torch.uint8
        )
        indices_random = bernoulli_mask & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(tokenizer), labels.shape, dtype=torch.long, device=inputs.device
        )
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels, indices_replaced, indices_random, masked_indices, labels_after_shift
