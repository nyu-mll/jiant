from string import punctuation


def _all_identical(items):
    for item in items:
        if item != items[0]:
            return False
    return True


def preterminal_subsumer(subsumees, current=None):
    if not current:
        if _all_identical(subsumees):
            return subsumees[0]
        else:
            current = subsumees[0].get_parent()

    for subsumee in subsumees:
        if subsumee not in current.preterminals():
            parent = current.get_parent()
            return preterminal_subsumer(subsumees, parent) if parent else None
    return current


class Node(object):
    def __init__(self, sexpression, parent):
        super(Node, self).__init__()
        self.sexpression = sexpression
        self.label = sexpression[0]
        self.parent = parent
        self._expansion = None
        self._preterminals = None

    def __str__(self):
        return str(self.sexpression)

    def expand(self):
        if self._expansion == None:
            self._expansion = []
            for child in self:
                self._expansion.append(child)
                if child.__class__ == Tree:
                    self._expansion.extend(child.expansion())
        return self._expansion

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def preterminals(self):
        if self._preterminals == None:
            self._preterminals = []
            for child in self:
                if child.__class__ == Preterminal:
                    self._preterminals.append(child)
                else:
                    self._preterminals.extend(child.preterminals())
        return self._preterminals

    # TODO rename: path_to_parent
    def path_to(self, target):
        if target == self:
            return self.label
        else:
            parent_path_to = self.parent.path_to(target) if self.parent else None
            return self.label + "/" + parent_path_to if parent_path_to else None

    # TODO rename: path_to
    def spath_to(self, target, direction="", path="", visited=None):
        if visited == None:
            visited = set()
        if self in visited:
            return None
        else:
            visited.add(self)
            if self == target:
                path = path + direction + target.label
                if self.__class__ == Preterminal:
                    path += "\\" + self.terminal
                return path

            else:
                if self.parent != None:
                    parent_path = self.parent.spath_to(target, "/", path, visited)
                    if parent_path != None:
                        path = path + direction + self.label + parent_path
                        if self.__class__ == Preterminal:
                            path = self.terminal + "/" + path
                        return path

                if self.__class__ != Preterminal:
                    for child in self:
                        child_path = child.spath_to(target, "\\", path, visited)
                        if child_path != None:
                            return path + direction + self.label + child_path

    def get_root(self):
        return self if self.parent == None else self.parent.get_root()


class Tree(Node, list):
    def __init__(self, sexpression, parent=None):
        list.__init__(self)
        Node.__init__(self, sexpression, parent)
        for child in sexpression[1:]:
            node_type = Preterminal if child.is_preterminal() else Tree
            self.append(node_type(child, self))

    def pretty(self, depth=0):
        string = ""
        if depth != 0:
            string += "\n"
        string += "  " * depth
        string += "(" + self.label
        for child in self:
            string += child.pretty(depth + 1)
        string += ")"
        return string

    def find_subsumer(self, subsumees):
        for subsumee in subsumees:
            if subsumee not in self.preterminals():
                return self.parent.find_subsumer(subsumees) if self.parent else None
        return self


class Preterminal(Node):
    def __init__(self, sexpression, parent):
        super(Preterminal, self).__init__(sexpression, parent)
        self.terminal = sexpression[1]

    def __str__(self):
        return "(%s %s)" % (self.label, self.terminal)

    def is_punctuation(self):
        return self.terminal[0] in punctuation

    def path_to(self, target):
        parent_path = super(Preterminal, self).path_to(target)
        return self.terminal + "/" + parent_path if parent_path else None

    def pretty(self, depth=0):
        indent = "  " * depth
        # return '\n%s(%s %s)' % (indent, self.label, self.terminal)
        return " (%s %s)" % (self.label, self.terminal)

    def preterminals(self):
        return [self]

    def find_subsumer(self, other_subsumees):

        all_match = True
        for other in other_subsumees:
            if not self == other:
                all_match = False

        if all_match:
            return self
        else:
            subsumees = [self]
            subsumees.extend(other_subsumees)
            return self.parent.find_subsumer(subsumees)
