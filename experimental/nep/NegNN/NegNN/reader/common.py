import codecs
import datetime
import os
import re
import shlex
import subprocess
import sys


def utf8_stdout():
    sys.stdout = codecs.getwriter("utf8")(sys.stdout)


def open(path, mode="r"):
    return codecs.open(path, mode, "utf-8")


def call(*command):
    subprocess.check_call(shlex.split(" ".join(command)))


def datestamp():
    now = datetime.datetime.now()
    return "%04d%02d%02d" % (now.year, now.month, now.day)


def to_path(*points):
    return "/".join(points)


class Folds(list):
    def __init__(self, items, n=10):
        super(Folds, self).__init__()
        for i in range(n):
            self.append(set())
        self.xref = dict()
        fold = 0
        for item in items:
            self[fold].add(item)
            self.xref[item.identifier()] = fold
            fold += 1
            if fold == len(self):
                fold = 0

    def get_fold(self, item):
        return self.xref[item.identifier()]


class Frequencies(dict):
    def __getitem__(self, key):
        return super(Frequencies, self).__getitem__(key) if key in self else 0

    def __setitem__(self, key, value):
        assert value.__class__ == int
        return super(Frequencies, self).__setitem__(key, value)

    def total(self):
        return sum(self.values())


class IdentifierXref(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = len(self)
        super(IdentifierXref, self).__getitem__(key)


class SExpression(list):
    def __init__(self, string, opener="(", closer=")", delimiter=" ", _position=0):
        super(SExpression, self).__init__()

        self._position = _position

        current_token = ""
        while self._position < len(string) - 1:

            self._position += 1

            character = string[self._position]

            if character == opener:
                if current_token:
                    self.append(current_token)
                    current_token = ""
                child = SExpression(string, opener, closer, delimiter, self._position)
                self.append(child)
                self._position = child._position

            elif character == closer:
                if current_token:
                    self.append(current_token)
                return

            elif character == delimiter:
                if current_token:
                    self.append(current_token)
                    current_token = ""

            else:
                current_token += character

    def __str__(self):
        return "(%s)" % " ".join([str(child) for child in self])

    def is_preterminal(self):
        for child in self:
            if child.__class__ == SExpression:
                return False
        return True


class Properties(dict):

    PATTERN = re.compile(r"\s*(.+?)\s*=\s*(.+)\s*")

    def __init__(self, path):
        super(Properties, self).__init__()
        self.path = path
        if os.path.exists(path):
            istream = open(path)
            for line in istream:
                line = line.strip()
                if len(line) > 0:
                    match = Properties.PATTERN.search(line)
                    key = match.group(1)
                    value = match.group(2)
                    if value.lower() in ("true", "false"):
                        value = bool(value)
                    self[key] = value
            istream.close()

    def commit(self):
        ostream = open(self.path, "w")
        for item in self.items():
            ostream.write("%s = %s\n" % item)
        ostream.close()


class Span(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return "(%d-%d)" % (self.start, self.end)

    def __len__(self):
        return self.end - self.start + 1

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def left_match(self, other):
        return self.start == other.start

    def right_match(self, other):
        return self.end == other.end

    def intersection(self, other):
        start = None
        end = None
        if self.start >= other.start and self.start <= other.end:
            start = self.start
            end = self.end if self.end < other.end else other.end
        elif self.end >= other.start and self.end <= other.end:
            start = self.start if self.start > other.start else other.start
            end = self.end
        return None if start == None and end == None else Span(start, end)

    def precision(self, other):
        intersection = self.intersection(other)
        return len(intersection) / float(len(self)) if intersection else float(0)

    def recall(self, other):
        intersection = self.intersection(other)
        return len(intersection) / float(len(other)) if intersection else float(0)
