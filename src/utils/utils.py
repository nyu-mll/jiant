"""
Assorted utilities for working with neural networks in AllenNLP.
"""
from typing import Dict, List, Sequence, Optional, Union, Iterable

import copy
import os
import json
import random
import logging
import codecs
import time

from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Dropout, Linear
from torch.nn import Parameter
from torch.nn import init

from allennlp.nn.util import masked_softmax, device_mapping
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common.params import Params


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


TOKENIZER = MosesTokenizer()
SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"

# Note: using the full 'detokenize()' method is not recommended, since it does
# a poor job of adding correct whitespace. Use unescape_xml() only.
_MOSES_DETOKENIZER = MosesDetokenizer()


def copy_iter(elems):
    '''Simple iterator yielding copies of elements.'''
    for elem in elems:
        yield copy.deepcopy(elem)


def wrap_singleton_string(item: Union[Sequence, str]):
    ''' Wrap a single string as a list. '''
    if isinstance(item, str):
        # Can't check if iterable, because a string is an iterable of
        # characters, which is not what we want.
        return [item]
    return item


def load_model_state(model, state_path, gpu_id, skip_task_models=[], strict=True):
    ''' Helper function to load a model state

    Parameters
    ----------
    model: The model object to populate with loaded parameters.
    state_path: The path to a model_state checkpoint.
    gpu_id: The GPU to use. -1 for no GPU.
    skip_task_models: If set, skip task-specific parameters for these tasks.
        This does not necessarily skip loading ELMo scalar weights, but I (Sam) sincerely
        doubt that this matters.
    strict: Whether we should fail if any parameters aren't found in the checkpoint. If false,
        there is a risk of leaving some parameters in their randomly initialized state.
    '''
    model_state = torch.load(state_path, map_location=device_mapping(gpu_id))

    assert_for_log(
        not (
            skip_task_models and strict),
        "Can't skip task models while also strictly loading task models. Something is wrong.")

    for name, param in model.named_parameters():
        # Make sure no trainable params are missing.
        if param.requires_grad:
            if strict:
                assert_for_log(name in model_state,
                               "In strict mode and failed to find at least one parameter: " + name)
            elif (name not in model_state) and ((not skip_task_models) or ("_mdl" not in name)):
                logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                logging.error("Parameter missing from checkpoint: " + name)
                logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if skip_task_models:
        keys_to_skip = []
        for task in skip_task_models:
            new_keys_to_skip = [key for key in model_state if "%s_mdl" % task in key]
            if new_keys_to_skip:
                logging.info("Skipping task-specific parameters for task: %s" % task)
                keys_to_skip += new_keys_to_skip
            else:
                logging.info("Found no task-specific parameters to skip for task: %s" % task)
        for key in keys_to_skip:
            del model_state[key]

    model.load_state_dict(model_state, strict=False)
    logging.info("Loaded model state from %s", state_path)


def get_elmo_mixing_weights(text_field_embedder, task=None):
    ''' Get pre-softmaxed mixing weights for ELMo from text_field_embedder for a given task.
    Stops program execution if something goes wrong (e.g. task is malformed, resulting in KeyError).

    args:
        - text_field_embedder (ElmoTextFieldEmbedder): the embedder used during the run
        - task (Task): a Task object with a populated `_classifier_name` attribute.

    returns:
        Dict[str, float]: dictionary with the values of each layer weight and of the scaling
                          factor.
    '''
    elmo = text_field_embedder.token_embedder_elmo._elmo
    if task:
        task_id = text_field_embedder.task_map[task._classifier_name]
    else:
        task_id = text_field_embedder.task_map["@pretrain@"]
    task_weights = getattr(elmo, "scalar_mix_%d" % task_id)
    params = {'layer%d' % layer_id: p.item() for layer_id, p in
              enumerate(task_weights.scalar_parameters.parameters())}
    params['gamma'] = task_weights.gamma
    return params


def get_batch_size(batch):
    ''' Given a batch with unknown text_fields, get an estimate of batch size '''
    batch_field = batch['inputs'] if 'inputs' in batch else batch['input1']
    keys = [k for k in batch_field.keys()]
    batch_size = batch_field[keys[0]].size()[0]
    return batch_size


def get_batch_utilization(batch_field, pad_idx=0):
    ''' Get ratio of batch elements that are padding

    Batch should be field, i.e. a dictionary of inputs'''
    if 'elmo' in batch_field:
        idxs = batch_field['elmo']
        pad_ratio = idxs.eq(pad_idx).sum().item() / idxs.nelement()
    else:
        raise NotImplementedError
    return 1 - pad_ratio


def maybe_make_dir(dirname):
    """Make a directory if it doesn't exist."""
    os.makedirs(dirname, exist_ok=True)


def unescape_moses(moses_tokens):
    '''Unescape Moses punctuation tokens.

    Replaces escape sequences like &#91; with the original characters
    (such as '['), so they better align to the original text.
    '''
    return [_MOSES_DETOKENIZER.unescape_xml(t) for t in moses_tokens]


def process_sentence(sent, max_seq_len, sos_tok=SOS_TOK, eos_tok=EOS_TOK):
    '''process a sentence '''
    max_seq_len -= 2
    assert max_seq_len > 0, "Max sequence length should be at least 2!"
    if isinstance(sent, str):
        return [sos_tok] + TOKENIZER.tokenize(sent)[:max_seq_len] + [eos_tok]
    elif isinstance(sent, list):
        assert isinstance(sent[0], str), "Invalid sentence found!"
        return [sos_tok] + sent[:max_seq_len] + [eos_tok]


def truncate(sents, max_seq_len, sos, eos):
    return [[sos] + s[:max_seq_len - 2] + [eos] for s in sents]


def load_json_data(filename: str) -> Iterable:
    ''' Load JSON records, one per line. '''
    with open(filename, 'r') as fd:
        for line in fd:
            yield json.loads(line)


def load_lines(filename: str) -> Iterable[str]:
    ''' Load text data, yielding each line. '''
    with open(filename) as fd:
        for line in fd:
            yield line.strip()


def load_diagnostic_tsv(
        data_file,
        max_seq_len,
        s1_idx=0,
        s2_idx=1,
        targ_idx=2,
        idx_idx=None,
        targ_map=None,
        targ_fn=None,
        skip_rows=0,
        delimiter='\t',
        filter_idx=None,
        filter_value=None):
    '''Load a tsv

    It loads the data with all it's attributes from diagnostic dataset for MNLI'''
    sent1s, sent2s, targs, idxs, lex_sem, pr_ar_str, logic, knowledge = [], [], [], [], [], [], [], []

    # There are 4 columns and every column could containd multiple values.
    # For every column there is a dict which maps index to (string) value and
    # different dict which maps value to index.
    ix_to_lex_sem_dic = {}
    ix_to_pr_ar_str_dic = {}
    ix_to_logic_dic = {}
    ix_to_knowledge_dic = {}

    lex_sem_to_ix_dic = {}
    pr_ar_str_to_ix_dic = {}
    logic_to_ix_dic = {}
    knowledge_to_ix_dic = {}

    # This converts tags to indices and adds new indices to dictionaries above.
    # In every row there could be multiple tags in one column
    def tags_to_ixs(tags, tag_to_ix_dict, ix_to_tag_dic):
        splitted_tags = tags.split(';')
        indexes = []
        for t in splitted_tags:
            if t == '':
                continue
            if t in tag_to_ix_dict:
                indexes.append(tag_to_ix_dict[t])
            else:
                # index 0 will be used for missing value
                highest_ix = len(tag_to_ix_dict)
                new_index = highest_ix + 1
                tag_to_ix_dict[t] = new_index
                ix_to_tag_dic[new_index] = t
                indexes.append(new_index)
        return indexes

    with codecs.open(data_file, 'r', 'utf-8', errors='ignore') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            try:
                row = row.rstrip().split(delimiter)
                sent1 = process_sentence(row[s1_idx], max_seq_len)
                if targ_map is not None:
                    targ = targ_map[row[targ_idx]]
                elif targ_fn is not None:
                    targ = targ_fn(row[targ_idx])
                else:
                    targ = int(row[targ_idx])
                sent2 = process_sentence(row[s2_idx], max_seq_len)
                sent2s.append(sent2)

                sent1s.append(sent1)
                targs.append(targ)

                lex_sem_sample = tags_to_ixs(row[0], lex_sem_to_ix_dic, ix_to_lex_sem_dic)
                pr_ar_str_sample = tags_to_ixs(row[1], pr_ar_str_to_ix_dic, ix_to_pr_ar_str_dic)
                logic_sample = tags_to_ixs(row[2], logic_to_ix_dic, ix_to_logic_dic)
                knowledge_sample = tags_to_ixs(row[3], knowledge_to_ix_dic, ix_to_knowledge_dic)

                idxs.append(row_idx)
                lex_sem.append(lex_sem_sample)
                pr_ar_str.append(pr_ar_str_sample)
                logic.append(logic_sample)
                knowledge.append(knowledge_sample)

            except Exception as e:
                print(e, " file: %s, row: %d" % (data_file, row_idx))
                continue

    ix_to_lex_sem_dic[0] = "missing"
    ix_to_pr_ar_str_dic[0] = "missing"
    ix_to_logic_dic[0] = "missing"
    ix_to_knowledge_dic[0] = "missing"

    lex_sem_to_ix_dic["missing"] = 0
    pr_ar_str_to_ix_dic["missing"] = 0
    logic_to_ix_dic["missing"] = 0
    knowledge_to_ix_dic["missing"] = 0

    return {'sents1': sent1s,
            'sents2': sent2s,
            'targs': targs,
            'idxs': idxs,
            'lex_sem': lex_sem,
            'pr_ar_str': pr_ar_str,
            'logic': logic,
            'knowledge': knowledge,
            'ix_to_lex_sem_dic': ix_to_lex_sem_dic,
            'ix_to_pr_ar_str_dic': ix_to_pr_ar_str_dic,
            'ix_to_logic_dic': ix_to_logic_dic,
            'ix_to_knowledge_dic': ix_to_knowledge_dic
            }


def load_tsv(
        data_file,
        max_seq_len,
        s1_idx=0,
        s2_idx=1,
        targ_idx=2,
        idx_idx=None,
        targ_map=None,
        targ_fn=None,
        skip_rows=0,
        delimiter='\t',
        filter_idx=None,
        filter_value=None):
    '''Load a tsv

    To load only rows that have a certain value for a certain column, like genre in MNLI, set filter_idx and filter_value.'''
    sent1s, sent2s, targs, idxs = [], [], [], []
    with codecs.open(data_file, 'r', 'utf-8', errors='ignore') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            try:
                row = row.strip().split(delimiter)
                if filter_idx and row[filter_idx] != filter_value:
                    continue
                sent1 = process_sentence(row[s1_idx], max_seq_len)
                if (targ_idx is not None and not row[targ_idx]) or not len(sent1):
                    continue

                if targ_idx is not None:
                    if targ_map is not None:
                        targ = targ_map[row[targ_idx]]
                    elif targ_fn is not None:
                        targ = targ_fn(row[targ_idx])
                    else:
                        targ = int(row[targ_idx])
                else:
                    targ = 0

                if s2_idx is not None:
                    sent2 = process_sentence(row[s2_idx], max_seq_len)
                    if not len(sent2):
                        continue
                    sent2s.append(sent2)

                if idx_idx is not None:
                    idx = int(row[idx_idx])
                    idxs.append(idx)

                sent1s.append(sent1)
                targs.append(targ)

            except Exception as e:
                print(e, " file: %s, row: %d" % (data_file, row_idx))
                continue

    if idx_idx is not None:
        return sent1s, sent2s, targs, idxs
    else:
        return sent1s, sent2s, targs


def split_data(data, ratio, shuffle=1):
    '''Split dataset according to ratio, larger split is first return'''
    n_exs = len(data[0])
    split_pt = int(n_exs * ratio)
    splits = [[], []]
    for col in data:
        splits[0].append(col[:split_pt])
        splits[1].append(col[split_pt:])
    return tuple(splits[0]), tuple(splits[1])

@Seq2SeqEncoder.register("masked_multi_head_self_attention")
class MaskedMultiHeadSelfAttention(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The dimension of the query and key projections which comprise the
        dot product attention function.
    values_dim : ``int``, required.
        The dimension which the input is projected to for representing the values,
        which are combined using the attention.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """

    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MaskedMultiHeadSelfAttention, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        self._query_projections = Parameter(torch.FloatTensor(num_heads, input_dim, attention_dim))
        self._key_projections = Parameter(torch.FloatTensor(num_heads, input_dim, attention_dim))
        self._value_projections = Parameter(torch.FloatTensor(num_heads, input_dim, values_dim))

        self._scale = input_dim ** 0.5
        self._output_projection = Linear(num_heads * values_dim,
                                         self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Because we are doing so many torch.bmm calls, which is fast but unstable,
        # it is critically important to intitialise the parameters correctly such
        # that these matrix multiplications are well conditioned initially.
        # Without this initialisation, this (non-deterministically) produces
        # NaNs and overflows.
        init.xavier_normal_(self._query_projections)
        init.xavier_normal_(self._key_projections)
        init.xavier_normal_(self._value_projections)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, timesteps, hidden_dim = inputs.size()
        if mask is None:
            mask = Variable(inputs.data.new(batch_size, timesteps).fill_(1.0))

        # Treat the queries, keys and values each as a ``num_heads`` size batch.
        # shape (num_heads, batch_size * timesteps, hidden_dim)
        inputs_per_head = inputs.repeat(num_heads, 1, 1).view(num_heads,
                                                              batch_size * timesteps,
                                                              hidden_dim)
        # Do the projections for all the heads at once.
        # Then reshape the result as though it had a
        # (num_heads * batch_size) sized batch.
        queries_per_head = torch.bmm(inputs_per_head, self._query_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        queries_per_head = queries_per_head.view(num_heads * batch_size,
                                                 timesteps,
                                                 self._attention_dim)

        keys_per_head = torch.bmm(inputs_per_head, self._key_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        keys_per_head = keys_per_head.view(num_heads * batch_size,
                                           timesteps,
                                           self._attention_dim)

        values_per_head = torch.bmm(inputs_per_head, self._value_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        values_per_head = values_per_head.view(num_heads * batch_size, timesteps, self._values_dim)

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = torch.bmm(
            queries_per_head, keys_per_head.transpose(
                1, 2)) / self._scale

        # Masking should go here
        causality_mask = subsequent_mask(timesteps).cuda()
        masked_scaled_similarities = scaled_similarities.masked_fill(causality_mask == 0, -1e9)

        # shape (num_heads * batch_size, timesteps, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        attention = masked_softmax(masked_scaled_similarities, mask.repeat(num_heads, 1))
        attention = self._attention_dropout(attention)
        # This is doing the following batch-wise matrix multiplication:
        # (num_heads * batch_size, timesteps, timesteps) *
        # (num_heads * batch_size, timesteps, values_dim)
        # which is equivalent to a weighted sum of the values with respect to
        # the attention distributions for each element in the num_heads * batch_size
        # dimension.
        # shape (num_heads * batch_size, timesteps, values_dim)
        outputs = torch.bmm(attention, values_per_head)

        # Reshape back to original shape (batch_size, timesteps, num_heads * values_dim)
        # Note that we _cannot_ use a reshape here, because this tensor was created
        # with num_heads being the first dimension, so reshaping naively would not
        # throw an error, but give an incorrect result.
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        return outputs

    @classmethod
    def from_params(cls, params: Params) -> 'MaskedMultiHeadSelfAttention':
        num_heads = params.pop_int('num_heads')
        input_dim = params.pop_int('input_dim')
        attention_dim = params.pop_int('attention_dim')
        values_dim = params.pop_int('values_dim')
        output_projection_dim = params.pop_int('output_projection_dim', None)
        attention_dropout_prob = params.pop_float('attention_dropout_prob', 0.1)
        params.assert_empty(cls.__name__)
        return cls(num_heads=num_heads,
                   input_dim=input_dim,
                   attention_dim=attention_dim,
                   values_dim=values_dim,
                   output_projection_dim=output_projection_dim,
                   attention_dropout_prob=attention_dropout_prob)


def assert_for_log(condition, error_message):
    assert condition, error_message


def check_arg_name(args):
    ''' Raise error if obsolete arg names are present. '''
    # Mapping - key: old name, value: new name
    name_dict = {'task_patience':'lr_patience',
                 'do_train': 'do_pretrain',
                 'train_for_eval':'do_target_task_training',
                 'do_eval': 'do_full_eval',
                 'train_tasks':'pretrain_tasks',
                 'eval_tasks':'target_tasks'}
    for old_name, new_name in name_dict.items():
        assert_for_log(old_name not in args,
                      "Error: Attempting to load old arg name [%s], please update to new name [%s]" %
                      (old_name,name_dict[old_name]))
