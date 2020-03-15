from . import common
from string import punctuation
from . import tree
import codecs


class _DataSetInventory(dict):
    def __init__(self, inventory):
        super(_DataSetInventory, self).__init__()
        for dataset_info in inventory:
            name, path = dataset_info
            dataset = None
            self[name] = (path, dataset)

    def get(self, name):
        if name not in self:
            raise ValueError("unknown dataset" + name)
        path, dataset = self[name]
        if not dataset:
            dataset = Data(path)
            self[name] = (path, dataset)
        return dataset


_dataset_inventory = _DataSetInventory(
    (
        ("development", "primary/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt"),
        ("training", "primary/SEM-2012-SharedTask-CD-SCO-training-09032012.txt"),
    )
)
# ('test-cardboard', 'primary/SEM-2012-SharedTask-CD-SCO-test-cardboard.txt'),
# ('test-circle', 'primary/SEM-2012-SharedTask-CD-SCO-test-circle.txt')))


def get_gold(name):
    return _dataset_inventory.get(name)


_cue_markers = "<>"
_scope_markers = "{}"
_event_markers = "||"

_bracket_escapes = {
    "(": "-LRB-",
    "{": "-LCB-",
    "[": "-LSB-",
    "]": "-RSB-",
    "}": "-RCB-",
    ")": "-RRB-",
}


def ptb_escape_brackets(string):
    if string in _bracket_escapes:
        return _bracket_escapes[string]
    else:
        return string


def tsdb_id(chapter, sentence):

    chapter_name = chapter[:-2]
    if chapter_name == "wisteria":
        tsdb_book = 10000
    elif chapter_name == "baskervilles":
        tsdb_book = 12000
    else:
        assert False, "unknown chapter " + chapter_name

    chapter_number = int(chapter[-2:])
    tsdb_chapter = (chapter_number - 1) * 1000

    return tsdb_book + tsdb_chapter + sentence


class Data(list):
    def __init__(self, path=None, istream=None):
        super(Data, self).__init__()

        if path:
            self.path = path
            istream = codecs.open(path, "rb", "utf8")
        elif istream:
            self.path = None
        else:
            raise ValueError("neither path nor istream specified")

        conll_sentence = []
        for line in istream:
            line = line.strip()
            # collect conll tokens until the entire
            # sentence has been read from file
            if line:
                conll_sentence.append(line.split("\t"))
            else:
                self.append(Sentence(conll_sentence))
                conll_sentence = []
        istream.close()
        # add the final sentence if file ended without blank line
        if conll_sentence:
            self.append(Sentence(conll_sentence))

        self.folds = common.Folds(self)

    def to_conll(self):
        return "\n\n".join([sentence.to_conll() for sentence in self])

    def find(self, chapter, number):
        for sentence in self:
            if sentence.chapter == chapter and sentence.number == number:
                return sentence

    def correct(self, corrections, deletions, flag):
        # get the sent index we are interested in correcting
        for i in corrections.iterkeys():
            ann_dict = corrections[i]
            # iterate through each token
            # check the annotation
            for t_i in range(len(self[i])):
                token = self[i][t_i]
                for a_i in range(len(token.get_annotations())):
                    if a_i in ann_dict.keys():
                        if flag == "cue":
                            token.annotations[a_i].cue = None
                        if flag == "event":
                            token.annotations[a_i].event = None
                        if t_i in ann_dict[a_i].keys():
                            # get word
                            w = ann_dict[a_i][t_i]
                            # reset annotation
                            if flag == "cue":
                                token.annotations[a_i].cue = w
                            if flag == "event":
                                token.annotations[a_i].event = w
        # delete those spurious annotations
        for i in deletions.iterkeys():
            ann_del = deletions[i]
            for t_i in self[i]:
                new_annotations = []
                for j in range(len(t_i.annotations)):
                    if j not in ann_del:
                        new_annotations.append(t_i.annotations[j])
                t_i.annotations = new_annotations


class Sentence(list):
    def __init__(self, conll_sentence):
        super(Sentence, self).__init__()
        self.number = int(conll_sentence[0][1])
        self.chapter = conll_sentence[0][0]
        for conll_token in conll_sentence:
            self.append(Token(self, conll_token))
        self.num_annotations = len(self[0].annotations) if self[0].annotations else 0
        self._tree = None

    def __str__(self):
        return " ".join([unicode(token) for token in self])

    def sent2tokens(self):
        return [unicode(token) for token in self]

    def __hash__(self):
        return self.identifier().__hash__()

    def _get_spans(self, annotation_id, accessor):
        spans = []
        start = None
        for token in self:
            annotation = accessor(token.annotations[annotation_id])
            if start == None and annotation:
                start = token.position
            elif start != None and not annotation:
                spans.append((start, token.position - 1))
                start = None
        if start != None:
            spans.append((start, token.position))
        return spans

    def identifier(self):
        return "%s %d" % (self.chapter, self.number)

    def get_num_annotations(self):
        return self.num_annotations

    def get_all_cues(self):
        return [[cs for cs in self.get_cues(a)] for a in range(self.num_annotations)]

    def get_all_events(self):
        return [[es for es in self.get_events(a)] for a in range(self.num_annotations)]

    def get_cues(self, annotation_id):
        return self._get_spans(annotation_id, Annotation.cue_accessor)

    def get_scopes(self, annotation_id):
        return self._get_spans(annotation_id, Annotation.scope_accessor)

    def get_events(self, annotation_id):
        return self._get_spans(annotation_id, Annotation.event_accessor)

    def get_full_scope(self):
        neg_instances = [[]] * self.num_annotations
        for tok in self:
            anns = tok.annotations
            for x in range(len(tok.annotations)):
                if anns[x].get_cue() != None:
                    neg_instances[x].append(anns[x].get_cue())
                if anns[x].get_event() != None:
                    neg_instances[x].append(anns[x].get_event())
                if anns[x].get_scope() != None:
                    neg_instances[x].append(anns[x].get_scope())
        return neg_instances

    def discontinuous_scope(self, annotation_id):
        cues = self.get_cues(annotation_id)
        scopes = self.get_scopes(annotation_id)
        for i in range(len(scopes) - 1):
            gap = (scopes[i][1] + 1, scopes[i + 1][0] - 1)
            if gap not in cues:
                return True
        return False

    def get_tree(self):
        if not self._tree:
            ptb = self.to_ptb()
            common.SExpression = common.SExpression(ptb)
            self._tree = tree.Tree(common.SExpression)
        return self._tree

    def to_conll(self):
        return "\n".join([token.to_conll() for token in self])

    def to_ptb(self):
        return "".join([token.to_ptb() for token in self])

    def pretty(self, annotation_id=None):

        if self.num_annotations == 0:
            return " ".join([token.word for token in self])

        strings = []

        if annotation_id != None:
            ids = [annotation_id]
        else:
            ids = range(self.num_annotations)

        in_cue = []
        in_scope = []
        in_event = []
        for i in range(max(ids) + 1):
            in_cue.append(False)
            in_scope.append(False)
            in_event.append(False)

        for i in range(len(self)):
            current = self[i]
            following = self[i + 1] if i < len(self) - 1 else None

            cue_starts = common.Frequencies()
            cue_ends = common.Frequencies()
            scope_starts = common.Frequencies()
            scope_ends = common.Frequencies()
            event_starts = common.Frequencies()
            event_ends = common.Frequencies()

            for identifier in ids:
                in_cue[identifier] = Token._indices(
                    in_cue[identifier],
                    current,
                    following,
                    identifier,
                    Annotation.cue_accessor,
                    cue_starts,
                    cue_ends,
                )
                in_scope[identifier] = Token._indices(
                    in_scope[identifier],
                    current,
                    following,
                    identifier,
                    Annotation.scope_accessor,
                    scope_starts,
                    scope_ends,
                )
                in_event[identifier] = Token._indices(
                    in_event[identifier],
                    current,
                    following,
                    identifier,
                    Annotation.event_accessor,
                    event_starts,
                    event_ends,
                )

            string = ""
            for i, char in enumerate(self[i].word):

                for j in range(scope_starts[i]):
                    string += _scope_markers[0]
                for j in range(event_starts[i]):
                    string += _event_markers[0]
                for j in range(cue_starts[i]):
                    string += _cue_markers[0]
                string += char
                for j in range(cue_ends[i]):
                    string += _cue_markers[1]
                for j in range(event_ends[i]):
                    string += _event_markers[1]
                for j in range(scope_ends[i]):
                    string += _scope_markers[1]
            strings.append(string)

        return " ".join(strings)

    """Added method"""

    def set_new_num_anns(self, new_num_annotations):
        self.num_annotations = new_num_annotations

    def unravel_neg_instance(self, trg_neg_instances):
        def decide(k, start):
            if k == "cue":
                ret = start + 0
            elif k == "event":
                ret = start + 1
            elif k == "scope":
                ret = start + 2
            return ret

        def substitute(indices, word, ann_string):
            for i in indices:
                ann_string[i] = word
            return ann_string

        start = 0
        dict_tokens = dict()
        print("TNI: ", len(trg_neg_instances))
        for instance in trg_neg_instances:
            instance_dict = instance.get_elementsAsDict()
            for k in instance_dict:
                for n in instance_dict[k]:
                    dict_tokens.setdefault(n, []).append(decide(k, start))
            start += 3
        for t in self:
            ann_string = ["_", "_", "_"] * len(trg_neg_instances)
            pos = t.get_position()
            if pos in dict_tokens:
                word = t.get_word()
                ann_string = substitute(dict_tokens[pos], word, ann_string)
            t.set_annotations(ann_string)


class Token(object):

    _no_annotation = "***"

    accessors = {
        "position": lambda token: token.cue,
        "word": lambda token: token.word,
        "lemma": lambda token: token.lemma,
    }

    def __init__(self, sentence, fields):
        self.sentence = sentence
        self.position = int(fields[2])
        self.word = fields[3]
        self.lemma = fields[4]
        self.pos = fields[5]
        self.syntax = fields[6]

        if len(fields) == 7 or fields[7] == Token._no_annotation:
            self.annotations = []
        else:
            self.annotations = [
                (Annotation(fields[i : i + 3])) for i in range(7, len(fields), 3)
            ]

    def __str__(self):
        return self.word

    def is_punctuation(self):
        return self.word[0] in punctuation

    def to_conll(self):
        fields = [
            self.sentence.chapter,
            str(self.sentence.number),
            str(self.position),
            self.word,
            self.lemma,
            self.pos,
            self.syntax,
        ]
        if self.annotations:
            for annotation in self.annotations:
                for field in annotation.fields():
                    fields.append(field)
        else:
            fields.append(Token._no_annotation)
        return "\t".join(fields)

    # Changed from (pos form) to (pos lemma) -- shouldn't cause a problem...?
    def to_ptb(self):
        return self.syntax.replace(
            "*",
            "(%s %s)"
            % (ptb_escape_brackets(self.pos), ptb_escape_brackets(self.lemma)),
        )

    @staticmethod
    def _indices(in_span, current, following, identifier, accessor, starts, ends):
        current_annotation = accessor(current.annotations[identifier])
        start = None
        if current_annotation:
            if not in_span:
                start = current.word.find(current_annotation)
                starts[start] += 1
                in_span = True
            if in_span:
                end_of_span = False

                if len(current_annotation) < len(current.word) and current.word.find(
                    current_annotation
                ) != len(current.word) - len(current_annotation):
                    end_of_span = True
                elif not following:
                    end_of_span = True
                else:
                    following_annotation = accessor(following.annotations[identifier])
                    if not following_annotation:
                        end_of_span = True
                    else:
                        if (
                            current.word.find(current_annotation)
                            + len(current_annotation)
                            < len(current.word) - 1
                        ):
                            end_of_span = True
                        elif (
                            following_annotation != following.word
                            and following.word.find(following_annotation) > 0
                        ):
                            end_of_span = True

                if end_of_span:
                    end = len(current_annotation) - 1
                    if start:
                        end += start
                    ends[end] += 1
                    in_span = False

        return in_span

    def get_word(self):
        return self.word

    def get_position(self):
        return self.position

    def get_annotations(self):
        return self.annotations

    def set_annotations(self, fields):
        self.annotations = [
            (Annotation(fields[i : i + 3])) for i in range(0, len(fields), 3)
        ]

    def is_cue(self):
        cues = [a.get_cue() for a in self.get_annotations()]
        return 0 if all(x == None for x in cues) else 1

    def is_event(self):
        events = [a.get_event() for a in self.get_annotations()]
        return 0 if all(x == None for x in events) else 1

    def is_scope(self):
        scope_els = [a.get_scope() for a in self.get_annotations()]
        return 0 if all(x == None for x in scope_els) else 1


class Annotation(object):

    _null = "_"

    cue_accessor = lambda annotation: annotation.cue
    scope_accessor = lambda annotation: annotation.scope
    event_accessor = lambda annotation: annotation.event

    def __init__(self, fields):
        self.cue = Annotation._in(fields[0])
        self.event = Annotation._in(fields[2])
        self.scope = Annotation._in(fields[1])

    def get_cue(self):
        return self.cue

    def get_event(self):
        return self.event

    def get_scope(self):
        return self.scope

    def get_elements_tuple(self):
        return (self.cue, self.event, self.scope)

    def fields(self):
        return [Annotation._out(field) for field in (self.cue, self.scope, self.event)]

    @staticmethod
    def _in(field):
        return None if field == Annotation._null else field

    @staticmethod
    def _out(field):
        return field if field else Annotation._null
