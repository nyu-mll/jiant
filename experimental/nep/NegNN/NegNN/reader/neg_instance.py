#!/usr/bin/env python

from itertools import izip
import pickle as pickle
import codecs

class ParNegationInstances(list):

    def __init__(self):
        super(ParNegationInstances,self).__init__()

    def gather_trg_side_negation(self,data_src,alignment_info,data_trg):
        i = 0
        for src_sent,wa_dict,trg_line in izip(data_src,alignment_info,data_trg):
            print "Processing sentence: ",i
            trg_neg_insts = list()
            for j in range(src_sent.num_annotations):
                trg_neg_insts.append(ParNegationInstance(j,src_sent,wa_dict))
            trg_line.unravel_neg_instance(trg_neg_insts)
            trg_line.num_annotations = len(trg_neg_insts)
            self.append(trg_neg_insts)
            i+=1

    def pickle_instances(self,pickle_file):
        pickle.dump(self,codecs.open(pickle_file,'wb','utf8'))

class ParNegationInstance:

    def __init__(self,index,src_sent,wa):
        # source sentence
        self.src_sent = src_sent
        # source spans
        # e.g [(14,14),(17,17)]
        self.src_cue_span = src_sent.get_cues(index)
        self.src_event_span = src_sent.get_events(index)
        self.src_scope_span = src_sent.get_scopes(index)

        # flat source spans
        # eg. [(14,14)] -->[14]
        self.flat_src_cue_span = self.get_flat_repr(self.src_cue_span)
        self.flat_src_event_span = self.get_flat_repr(self.src_event_span)

        # source spans from original manual annotations(that might include source spans)
        self.org_cue_span = self.get_wa_span(self.src_cue_span,wa)
        self.org_event_span = self.get_wa_span(self.src_event_span,wa)
        self.org_scope_span = self.get_wa_span(self.src_scope_span,wa)

        # get flat representation of the original source span
        self.flat_org_cue_span = self.get_flat_repr(self.org_cue_span)
        self.flat_org_event_span = self.get_flat_repr(self.org_event_span)

        # cue or event index relative to the newly created sub-string
        self.neg_el_index_cue = []
        self.neg_el_index_event = []

        # source words
        self.src_cue_w = self.get_words_from_spans(src_sent,self.flat_org_cue_span,self.flat_src_cue_span,"c")
        self.src_event_w = self.get_words_from_spans(src_sent,self.flat_org_event_span,self.flat_src_event_span,"e")
        self.src_scope_w = None

        # target indices
        self.trg_cue_i = []
        self.trg_event_i = []
        self.trg_scope_i = []
        # src_alignment: one or many or None (if empty)
        self._from_cue = "n"
        self._from_event = "n"
        # trg_alignment: one or many or None (if empty)
        self._to_cue = "n"
        self._to_event = "n"

        # fill target negation
        self.create_trg_negation(wa)
        # fill alignment information
        self.alignment_type()
        # print the content of the parallel instance for safety check
        self.safety_check()

    def get_elementsAsDict(self):
        return {'cue':self.trg_cue_i,
        'event':self.trg_event_i,
        'scope':self.trg_scope_i}

    # get full list representation instead of tuples
    def get_flat_repr(self,spans):
        single_indeces = set()
        for s in spans:
            if len(s)==1: s=(s[0],s[0])
            for k in range(s[0],s[-1]+1):
                single_indeces.add(k)
        return sorted(list(single_indeces))

    # gets original word alignment span
    def get_wa_span(self,spans,wa):
        ann_phrases = []
        for s in spans:
            if len(s)==1: s=(s[0],s[0])
            for k in range(s[0],s[-1]+1):
                if k in wa:
                    ann_phrases.extend([wa[k].keys()[0]])
        return ann_phrases

    def get_words_from_spans(self,src_sent,org_indices,trg_indices,flag):
        r = list()
        for n in org_indices:
            r.append(src_sent[n].word)
            if n in trg_indices:
                if flag=="c":
                    self.neg_el_index_cue.append(org_indices.index(n))
                elif flag=="e":
                    self.neg_el_index_event.append(org_indices.index(n))
        return r

    def create_trg_negation(self,wa):
        # given a list of sr side spans, return a list target side indeces
        self.trg_cue_i = sorted(self.get_trg_indices(self.src_cue_span,wa))
        self.trg_event_i = sorted(self.get_trg_indices(self.src_event_span,wa))
        self.trg_scope_i = sorted(self.get_trg_indices(self.src_scope_span,wa))
        
    def get_trg_indices(self,src_tuples,wa_dict):
            src_indices,trg_indices = list(),set()
            for _tuple in src_tuples:
                src_indices.extend(range(_tuple[0],_tuple[1]+1))
            for s_i in src_indices:
                if wa_dict.has_key(s_i):
                    trg_indices |= set([i for nl in wa_dict[s_i].values()[0] for i in nl])
            return list(trg_indices)

    def alignment_type(self):
        self._from_cue = "o" if len(self.flat_org_cue_span)==1 else "m"
        self._to_cue = "o" if len(self.trg_cue_i)==1 else "m"
        self._from_event = "o" if len(self.flat_org_event_span)==1 else "m"
        self._to_event = "o" if len(self.trg_event_i)==1 else "m"

    def safety_check(self):
        print "********************************"
        print "**********PRINTINT PNI**********"
        print "Src_cue_span: ",self.src_cue_span
        print "Src_event_span: ", self.src_event_span
        print "Src_scope_span: ",self.src_scope_span

        # flat source spans
        # eg. [(14,14)] -->[14]
        print "flat_src_cue_span: ",self.flat_src_cue_span
        print "flat_src_event_span: ",self.flat_src_event_span

        # source spans from original manual annotations(that might include source spans)
        print "org_cue_span",self.org_cue_span
        print "org_event_span",self.org_event_span
        print "org_scope_span",self.org_scope_span

        # get flat representation of the original source span
        print "flat_org_cue_span",self.flat_org_cue_span
        print "flat_org_event_span",self.flat_org_event_span

        # source words
        print "src_cue_w",self.src_cue_w
        print "src_event_w",self.src_event_w
        print "src_scope_w",self.src_scope_w
        
        # cue or event index relative to the newly created sub-string
        print "neg_el_index_cue",self.neg_el_index_cue
        print "neg_el_index_event",self.neg_el_index_event

        # target indices
        print "trg_cue_i",self.trg_cue_i
        print "trg_event_i",self.trg_event_i
        print "trg_scope_i",self.trg_scope_i
        # src_alignment: one or many or None (if empty)
        print "_from_cue",self._from_cue
        print "_from_event",self._from_event
        # trg_alignment: one or many or None (if empty)
        print "_to_cue",self._to_cue
        print "_to_event",self._to_event
        print "*******************************************\n"
