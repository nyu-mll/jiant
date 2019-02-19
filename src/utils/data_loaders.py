"""
Functions having to do with loading data from output of
files downloaded in scripts/download_data_glue.py

"""
from .tokenizers import AVAILABLE_TOKENIZERS
import codecs

SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"

def load_tsv(
        tokenizer_name,
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
        filter_value=None,):
    '''Load a tsv

    To load only rows that have a certain value for a certain column, like genre in MNLI, set filter_idx and filter_value.'''
    sent1s, sent2s, targs, idxs = [], [], [], []
    with codecs.open(data_file, 'r', 'utf-8', errors='ignore') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            row = row.strip().split(delimiter)
            if filter_idx and row[filter_idx] != filter_value:
                continue
            sent1 = process_sentence(tokenizer_name, row[s1_idx], max_seq_len)
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
                sent2 = process_sentence(tokenizer_name, row[s2_idx], max_seq_len)
                if not len(sent2):
                    continue
                sent2s.append(sent2)

            if idx_idx is not None:
                idx = int(row[idx_idx])
                idxs.append(idx)

            sent1s.append(sent1)
            targs.append(targ)

    if idx_idx is not None:
        return sent1s, sent2s, targs, idxs
    else:
        return sent1s, sent2s, targs

def load_diagnostic_tsv(
        tokenizer_name,
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
            row = row.rstrip().split(delimiter)
            sent1 = process_sentence(tokenizer_name, row[s1_idx], max_seq_len)
            if targ_map is not None:
                targ = targ_map[row[targ_idx]]
            elif targ_fn is not None:
                targ = targ_fn(row[targ_idx])
            else:
                targ = int(row[targ_idx])
            sent2 = process_sentence(tokenizer_name, row[s2_idx], max_seq_len)
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

def process_sentence(tokenizer_name, sent, max_seq_len, sos_tok=SOS_TOK, eos_tok=EOS_TOK):
    '''process a sentence '''
    max_seq_len -= 2
    assert max_seq_len > 0, "Max sequence length should be at least 2!"
    TOKENIZER = AVAILABLE_TOKENIZERS[tokenizer_name]
    if isinstance(sent, str):
        return [sos_tok] + TOKENIZER.tokenize(sent)[:max_seq_len] + [eos_tok]
    elif isinstance(sent, list):
        assert isinstance(sent[0], str), "Invalid sentence found!"
        return [sos_tok] + sent[:max_seq_len] + [eos_tok]
