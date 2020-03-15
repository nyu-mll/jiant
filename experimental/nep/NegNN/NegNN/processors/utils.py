# -*-coding:utf-8-*-
#! /usr/bin/env python

import codecs

# try different data formats
def data2sents(sets, look_event, look_scope, lang):
    def get_uni_mapping(lang):
        mapping = {}
        f = codecs.open(
            "./NegNN/NegNN/data/uni_pos_map/%s.txt" % lang, "rb", "utf8"
        ).readlines()
        for line in f:
            spl = line.strip().split("\t")
            _pos = spl[0].split("|")[0]
            mapping.update({_pos: spl[1]})
        return mapping

    def segment(word, is_cue):
        _prefix_one = ["a"]
        _prefix_two = ["ab", "un", "im", "in", "ir", "il"]
        _prefix_three = ["dis", "non"]
        _suffix = ["less", "lessness", "lessly"]

        which_suff = [word.endswith(x) for x in _suffix]

        if is_cue:
            if word.lower()[:2] in _prefix_two and len(word) > 4:
                return ([word[:2] + "-", word[2:]], 1)
            elif word.lower()[:1] in _prefix_one and len(word) > 4:
                return ([word[:1] + "-", word[1:]], 1)
            elif word.lower()[:3] in _prefix_three and len(word) > 4:
                return ([word[:3] + "-", word[3:]], 1)
            elif True in which_suff and len(word) > 4:
                idx = [i for i, v in enumerate(which_suff) if v][0]
                suff_cue = _suffix[idx]
                return ([word[: -len(suff_cue)], "-" + suff_cue], 0)
            else:
                return ([word], None)

        else:
            return ([word], None)

    def assign_tag(is_event, is_scope, look_event, look_scope):
        if is_event and look_event:
            return "E"
        elif is_scope and look_scope:
            return "I"
        else:
            return "O"

    sents = []
    tag_sents = []
    ys = []
    lengths = []

    cues_one_hot = []
    scopes_one_hot = []

    for d in sets:
        length = 0
        for s_idx, s in enumerate(d):
            all_cues = [
                i
                for i in range(len(s))
                if filter(lambda x: x.cue != None, s[i].annotations) != []
            ]
            if len(s[0].annotations) > 0:
                for curr_ann in range(len(s[0].annotations)):
                    cues_idxs = [
                        i[0]
                        for i in filter(
                            lambda x: x[1] != None,
                            [
                                (i, s[i].annotations[curr_ann].cue)
                                for i in range(len(s))
                            ],
                        )
                    ]
                    event_idxs = [
                        i[0]
                        for i in filter(
                            lambda x: x[1] != None,
                            [
                                (i, s[i].annotations[curr_ann].event)
                                for i in range(len(s))
                            ],
                        )
                    ]
                    scope_idxs = [
                        i[0]
                        for i in filter(
                            lambda x: x[1] != None,
                            [
                                (i, s[i].annotations[curr_ann].scope)
                                for i in range(len(s))
                            ],
                        )
                    ]

                    sent = []
                    tag_sent = []
                    y = []

                    cue_one_hot = []
                    scope_one_hot = []

                    for t_idx, t in enumerate(s):
                        word, tag = t.word, t.pos
                        word_spl, word_idx = segment(word, t_idx in all_cues)
                        if len(word_spl) == 1:
                            _y = assign_tag(
                                t_idx in event_idxs,
                                t_idx in scope_idxs,
                                look_event,
                                look_scope,
                            )
                            c_info = ["NOTCUE"] if t_idx not in cues_idxs else ["CUE"]
                            s_info = ["S"] if t_idx in scope_idxs else ["NS"]
                            tag_info = [tag]

                        elif len(word_spl) == 2:
                            _y_word = assign_tag(
                                t_idx in event_idxs,
                                t_idx in scope_idxs,
                                look_event,
                                look_scope,
                            )
                            if t_idx in cues_idxs:
                                _y = [_y_word, "O"] if word_idx == 0 else ["O", _y_word]
                                c_info = (
                                    ["NOTCUE", "CUE"]
                                    if word_idx == 0
                                    else ["CUE", "NOTCUE"]
                                )
                                s_info = ["S", "NS"] if word_idx == 0 else ["NS", "S"]
                            else:
                                _y = [_y_word, _y_word]
                                c_info = ["NOTCUE", "NOTCUE"]
                                s_info = (
                                    ["S", "S"] if t_idx in scope_idxs else ["NS", "NS"]
                                )
                            tag_info = [tag, "AFF"] if word_idx == 0 else ["AFF", tag]
                        # add the word(s) to the sentence list
                        sent.extend(word_spl)
                        # add the POS tag(s) to the TAG sentence list
                        tag_sent.extend(tag_info)
                        # add the _y for the word
                        y.extend(_y)
                        # extend the cue hot vector
                        cue_one_hot.extend(c_info)
                        # extend the scope hot vector
                        scope_one_hot.extend(s_info)

                    sents.append(sent)
                    tag_sents.append(tag_sent)
                    ys.append(y)
                    cues_one_hot.append(cue_one_hot)
                    scopes_one_hot.append(scope_one_hot)
                    length += 1

        lengths.append(length)
    # make normal POS tag into uni POS tags
    pos2uni = get_uni_mapping(lang)
    tag_uni_sents = [[pos2uni[t] for t in _s] for _s in tag_sents]

    return sents, tag_sents, tag_uni_sents, ys, cues_one_hot, scopes_one_hot, lengths
