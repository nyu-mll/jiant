import argparse
import re

#import matplotlib.pyplot as plt
#import nltk
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import \
    SingleIdTokenIndexer, ELMoTokenCharactersIndexer, \
    TokenCharactersIndexer
import os
import sys

sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../../"))
from src.preprocess import build_tasks
from src.utils.utils import assert_for_log, maybe_make_dir, load_model_state
from src.models import build_model
from src.utils import config
import glob
import data
import nltk
#import data
#import data_ptb
#from utils import batchify, get_batch, repackage_hidden

#from parse_comparison import corpus_stats_labeled, corpus_average_depth
#from data_ptb import word_tags
word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
criterion = nn.CrossEntropyLoss()
def evaluate(data_source, batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output = model.decoder(output)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)

def corpus2idx(sentence):
    arr = np.array([data.dictionary.word2idx[c] for c in sentence.split()], dtype=np.int32)
    return torch.from_numpy(arr[:, None]).long()


# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


# def build_tree(depth, sen):
#     assert len(depth) == len(sen)
#     assert len(depth) >= 0
#
#     if len(depth) == 1:
#         parse_tree = sen[0]
#     else:
#         idx_max = numpy.argmax(depth[1:]) + 1
#         parse_tree = []
#         if len(sen[:idx_max]) > 0:
#             tree0 = build_tree(depth[:idx_max], sen[:idx_max])
#             parse_tree.append(tree0)
#         if len(sen[idx_max:]) > 0:
#             tree1 = build_tree(depth[idx_max:], sen[idx_max:])
#             parse_tree.append(tree1)
#     return parse_tree


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1

def MRG(tr):
    if isinstance(tr, str):
        #return '(' + tr + ')'
        return tr + ' '
    else:
        s = '( '
        for subtr in tr:
            s += MRG(subtr)
        s += ') '
        return s

def MRG_labeled(tr):
    if isinstance(tr, nltk.Tree):
        if tr.label() in word_tags:
            return tr.leaves()[0] + ' '
        else:
            s = '(%s ' % (re.split(r'[-=]', tr.label())[0])
            for subtr in tr:
                s += MRG_labeled(subtr)
            s += ') '
            return s
    else:
        return ''

def mean(x):
    return sum(x) / len(x)


def test(model, corpus, cuda, prt=False):
    model.eval()

    prec_list = []
    reca_list = []
    f1_list = []

    nsens = 0
    word2idx = corpus.dictionary.word2idx
    if args.wsj10:
        dataset = zip(corpus.train_sens, corpus.train_trees, corpus.train_nltktrees)
    else:
        dataset = zip(corpus.test_sens, corpus.test_trees, corpus.test_nltktrees)

    corpus_sys = {}
    corpus_ref = {}
    for sen, sen_tree, sen_nltktree in dataset:
        if args.wsj10 and len(sen) > 12:
            continue    
        x = numpy.array([word2idx[w] if w in word2idx else word2idx['<unk>'] for w in sen])
        input = Variable(torch.LongTensor(x[:, None]))
        if cuda:
            input = input.cuda()

        hidden = model.init_hidden(1)
        _, hidden = model(input, hidden)

        distance = model.distance[0].squeeze().data.cpu().numpy()
        distance_in = model.distance[1].squeeze().data.cpu().numpy()

        nsens += 1
        if prt and nsens % 100 == 0:
            for i in range(len(sen)):
                print('%15s\t%s\t%s' % (sen[i], str(distance[:, i]), str(distance_in[:, i])))
            print('Standard output:', sen_tree)

        sen_cut = sen[1:-1]
        # gates = distance.mean(axis=0)
        for gates in [
            # distance[0],
            distance[1],
            # distance[2],
            # distance.mean(axis=0)
        ]:
            depth = gates[1:-1]
            parse_tree = build_tree(depth, sen_cut)

            corpus_sys[nsens] = MRG(parse_tree)
            corpus_ref[nsens] = MRG_labeled(sen_nltktree)

            model_out, _ = get_brackets(parse_tree)
            std_out, _ = get_brackets(sen_tree)
            overlap = model_out.intersection(std_out)

            prec = float(len(overlap)) / (len(model_out) + 1e-8)
            reca = float(len(overlap)) / (len(std_out) + 1e-8)
            if len(std_out) == 0:
                reca = 1.
                if len(model_out) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            prec_list.append(prec)
            reca_list.append(reca)
            f1_list.append(f1)

            if prt and nsens % 100 == 0:
                print('Model output:', parse_tree)
                print('Prec: %f, Reca: %f, F1: %f' % (prec, reca, f1))

        if prt and nsens % 100 == 0:
            print('-' * 80)

            f, axarr = plt.subplots(3, sharex=True, figsize=(distance.shape[1] // 2, 6))
            axarr[0].bar(numpy.arange(distance.shape[1])-0.2, distance[0], width=0.4)
            axarr[0].bar(numpy.arange(distance_in.shape[1])+0.2, distance_in[0], width=0.4)
            axarr[0].set_ylim([0., 1.])
            axarr[0].set_ylabel('1st layer')
            axarr[1].bar(numpy.arange(distance.shape[1]) - 0.2, distance[1], width=0.4)
            axarr[1].bar(numpy.arange(distance_in.shape[1]) + 0.2, distance_in[1], width=0.4)
            axarr[1].set_ylim([0., 1.])
            axarr[1].set_ylabel('2nd layer')
            axarr[2].bar(numpy.arange(distance.shape[1]) - 0.2, distance[2], width=0.4)
            axarr[2].bar(numpy.arange(distance_in.shape[1]) + 0.2, distance_in[2], width=0.4)
            axarr[2].set_ylim([0., 1.])
            axarr[2].set_ylabel('3rd layer')
            plt.sca(axarr[2])
            plt.xlim(xmin=-0.5, xmax=distance.shape[1] - 0.5)
            plt.xticks(numpy.arange(distance.shape[1]), sen, fontsize=10, rotation=45)

            plt.savefig('figure/%d.png' % (nsens))
            plt.close()

    prec_list, reca_list, f1_list \
        = numpy.array(prec_list).reshape((-1,1)), numpy.array(reca_list).reshape((-1,1)), numpy.array(f1_list).reshape((-1,1))
    if prt:
        print('-' * 80)
        numpy.set_printoptions(precision=4)
        print('Mean Prec:', prec_list.mean(axis=0),
              ', Mean Reca:', reca_list.mean(axis=0),
              ', Mean F1:', f1_list.mean(axis=0))
        print('Number of sentence: %i' % nsens)

        correct, total = corpus_stats_labeled(corpus_sys, corpus_ref)
        print(correct)
        print(total)
        print('ADJP:', correct['ADJP'], total['ADJP'])
        print('NP:', correct['NP'], total['NP'])
        print('PP:', correct['PP'], total['PP'])
        print('INTJ:', correct['INTJ'], total['INTJ'])
        print(corpus_average_depth(corpus_sys))

    return f1_list.mean(axis=0)


if __name__ == '__main__':
    marks = [' ', '-', '=']

    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)
    #python scripts/parsing/test_phrase_grammar.py --exp_dir ptb.th
    parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    parser.add_argument('--data', type=str, default='data/penn',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='PTB.pt',
                            help='onlstmconfig/test1_15_75/model_state_main_epoch_269.best_macro.th')
    parser.add_argument('--exp_dir', type=str, default='PTB.pt',
                            help='model checkpoint to use')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--wsj10', action='store_true',
                        help='use WSJ10')
    # parser.add_argument('--pretrain_tasks', type=str, default='',
    #                     help='use WSJ10')
    # parser.add_argument('--target_tasks', type=str, default='wsj',
    #                     help='use WSJ10')
    # parser.add_argument('--max_seq_len', type=int, default=70,
    #                     help='use WSJ10')
    # parser.add_argument('--reload_tasks', type=bool, default=70,
    #                     help='reload tasks')
    parser.add_argument('--data_dir', type=str, default="data/",
                        help='reload tasks')

    parser.add_argument('--max_targ_word_v_size', type=int, default=20000,
                        help='maxt')

    parser.add_argument('--config_file', type=str, default='',
                        help='maxt')
    parser.add_argument('--ptb_path', type=str, default='/Users/anhadmohananey/Downloads/ptb_sec23.jsonl')
    args = parser.parse_args()
    clargs=config.params_from_file(args.config_file, None)
    torch.manual_seed(args.seed)
    pretrain_tasks, target_tasks, vocab, word_embs=build_tasks(clargs)
    tasks = sorted(set(pretrain_tasks + target_tasks), key=lambda x: x.name)
    model = build_model(clargs, vocab, word_embs, tasks)

    macro_best = glob.glob(os.path.join(clargs.run_dir,
                                                "model_state_main_epoch*"))
    load_model_state(model,macro_best[-1],args.cuda)
    corpus=data.Corpus(vocab._token_to_index['tokens'])
    f1_list=[]
    prec_list=[]
    reca_list=[]
    f1_list=[]
    model.eval()
    for i in range(len(corpus.test)):
        st=corpus.test[i].reshape(1, -1)
        ta=torch.cat([corpus.test[i][1:],corpus.test[i][:1]]).reshape(1,-1)
        inp={}
        tmp={}
        tmp['words']=st
        inp['input']=tmp
        tmp1={}
        tmp1['words']=ta
        inp['targs']=tmp1
        inp['targs_b']=tmp1
        output=model.forward(tasks[0],inp)
        distances=model.sent_encoder._phrase_layer.distances
        dc=distances[1][1:-1]
        sen_cut=corpus.test[i][1:-1]
        sen_tree=corpus.test_trees[i]
        parse_tree=build_tree(dc.detach(), sen_cut)
        model_out, _ = get_brackets(parse_tree)
        std_out, _ = get_brackets(sen_tree)
        overlap = model_out.intersection(std_out)
        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)
        if len(std_out) == 0:
            reca = 1.
            if len(model_out) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        print(f1)
        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1)

        if i%100==0:
            print("-----")
            print(str(i)+"___")
            print(mean(f1_list))
            print("----")