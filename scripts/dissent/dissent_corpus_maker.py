import re
import sys

wiki = open(sys.argv[1], "r")
fo = open(sys.argv[2], "w")


word_list = []
parent_list = []
rel_list = []

marked = 0
root_index = -1

sent_counter = 0


def alphanum_core(string):
    words = string.split()

    seen_letters = 0
    sentence_new = []
    for word in words:
        if any(c.isalpha() for c in word):  # re.search('[a-zA-Z0-9]', word) == 1:
            seen_letters = 1
        if seen_letters:
            sentence_new.append(word)

    final_has_letters = 0
    for word_ind, word in enumerate(sentence_new):
        if any(c.isalpha() for c in word):  # re.search('[a-zA-Z0-9]', word) == 1:
            final_has_letters = word_ind

    return " ".join(sentence_new[: final_has_letters + 1])


def external_punc_remover(string):
    words = string.split()
    if len(words) == 0:
        return string

    if words[0] in [".", ",", "/", ":", ";", "?", "!"]:
        return external_punc_remover(" ".join(words[1:]))
    if words[-1] in [".", ",", "/", ":", ";", "?", "!"]:
        return external_punc_remover(" ".join(words[:-1]))

    if words[0] == "'" and "'" not in alphanum_core(string):
        return external_punc_remover(" ".join(words[1:]))
    if words[-1] == "'" and "'" not in alphanum_core(string):
        return external_punc_remover(" ".join(words[:-1]))
    if words[0] == '"' and '"' not in alphanum_core(string):
        return external_punc_remover(" ".join(words[1:]))
    if words[-1] == '"' and '"' not in alphanum_core(string):
        return external_punc_remover(" ".join(words[:-1]))

    if words[0] == "''" and "''" not in alphanum_core(string):
        return external_punc_remover(" ".join(words[1:]))
    if words[-1] == "''" and "''" not in alphanum_core(string):
        return external_punc_remover(" ".join(words[:-1]))

    return string


def parent_function(
    parent_list, this_index, desired_parent, rel_list, opaque_index_1, opaque_index_2
):
    if this_index == desired_parent:
        return -1
    # or rel_list[this_index] == "conj" or rel_list[this_index] == "advcl" or
    # rel_list[this_index] == "punct" or rel_list[this_index] == "cc":
    if parent_list[this_index] == 0 or this_index == opaque_index_1 or this_index == opaque_index_2:
        return 0
    elif parent_list[this_index] == desired_parent + 1:
        return -1
    else:
        return parent_list[this_index]


def get_all_descendants(word_list, parent_list, index, rel_list, opaque_index_1, opaque_index_2):
    binary = []
    for this_index, parent in enumerate(parent_list):
        done = 0
        while not done:
            this_par = parent_function(
                parent_list, this_index, index, rel_list, opaque_index_1, opaque_index_2
            )
            if this_par == 0 or this_par == -1:
                in_out = this_par * -1
                binary.append(in_out)
                done = 1
            else:
                this_index = this_par - 1

    connected = 0
    ended = 0

    checked_r = 0
    to_add = 0
    seen_zero = 0
    while not checked_r:
        this_digit = binary[index + to_add]
        if this_digit == 0:
            seen_zero = 1
        if seen_zero:
            binary[index + to_add] = 0
        if index + to_add == len(binary) - 1:
            checked_r = 1
        to_add += 1
    checked_l = 0
    to_add = 0
    seen_zero = 0
    while not checked_l:
        this_digit = binary[index - to_add]
        if this_digit == 0:
            seen_zero = 1
        if seen_zero:
            binary[index - to_add] = 0
        if index - to_add == 0:
            checked_l = 1
        to_add += 1

    for bin_index, digit in enumerate(binary):
        if digit == 1:
            connected = 1
        if connected and digit == 0 and rel_list[bin_index] == "punct":
            binary[bin_index] = 1
        elif connected and digit == 0:
            ended = 1
        if ended and digit == 1:
            connected = 0

    if not connected:
        return -1

    sentence = []
    started = 0
    for index, value in enumerate(binary):
        if value == 1:
            if started == 0:
                started = 1
                if rel_list[index] != "mark":
                    sentence.append(word_list[index])
            else:
                sentence.append(word_list[index])

    return " ".join(sentence)

    seen_letters = 0
    sentence_new = []
    for word in sentence:
        if any(c.isalpha() for c in word):  # re.search('[a-zA-Z0-9]', word) == 1:
            seen_letters = 1
        if seen_letters:
            sentence_new.append(word)

    final_has_letters = 0
    for word_ind, word in enumerate(sentence_new):
        if any(c.isalpha() for c in word):  # re.search('[a-zA-Z0-9]', word) == 1:
            final_has_letters = word_ind

    return " ".join(sentence_new[: final_has_letters + 1])


def has_subj(rel_list, parent_list, index):

    for rel_index, rel in enumerate(rel_list):
        if rel == "nsubj" and parent_list[rel_index] == index + 1:
            return 1

    return 0


prev_sent = ""
for line in wiki:
    sent_counter += 1
    if sent_counter % 100000 == 0:
        print(sent_counter)
        if sent_counter == 0:
            print(sys.argv[1])

    if len(line) <= 2:

        if marked == 1:
            for index, word in enumerate(word_list):
                if rel_list[index] == "root":
                    root_index = index

            for index, word in enumerate(word_list):

                if (
                    word.lower() == "because"
                    or word.lower() == "if"
                    or word.lower() == "before"
                    or word.lower() == "so"
                    or word.lower() == "though"
                ):
                    if rel_list[index] == "mark" or rel_list[index] == "advmod":
                        parent = parent_list[index] - 1
                        grandparent = parent_list[parent] - 1
                        if (
                            rel_list[parent] == "advcl"
                            and has_subj(rel_list, parent_list, grandparent)
                            and has_subj(rel_list, parent_list, parent)
                        ):
                            sent1 = get_all_descendants(
                                word_list, parent_list, grandparent, rel_list, index, parent
                            )
                            sent2 = get_all_descendants(
                                word_list, parent_list, parent, rel_list, index, index
                            )
                            connector = word
                            fo.write(
                                external_punc_remover(sent1)
                                + "\t"
                                + external_punc_remover(sent2)
                                + "\t"
                                + connector
                                + "\n"
                            )
                        elif grandparent == -1:
                            if index == 0:
                                sent1 = prev_sent
                                sent2 = " ".join(word_list[1:])
                                connector = word
                                fo.write(
                                    external_punc_remover(sent1)
                                    + "\t"
                                    + external_punc_remover(sent2)
                                    + "\t"
                                    + connector
                                    + "\n"
                                )
                            # pass # Here is where it would be previous sentence
                    else:
                        continue
                if word.lower() == "when":
                    if rel_list[index] == "advmod":
                        parent = parent_list[index] - 1
                        grandparent = parent_list[parent] - 1
                        if (
                            rel_list[parent] == "advcl"
                            and has_subj(rel_list, parent_list, grandparent)
                            and has_subj(rel_list, parent_list, parent)
                        ):
                            sent1 = get_all_descendants(
                                word_list, parent_list, grandparent, rel_list, index, parent
                            )
                            sent2 = get_all_descendants(
                                word_list, parent_list, parent, rel_list, index, index
                            )
                            connector = word
                            fo.write(
                                external_punc_remover(sent1)
                                + "\t"
                                + external_punc_remover(sent2)
                                + "\t"
                                + connector
                                + "\n"
                            )
                        elif False:  # grandparent == -1:
                            if index == 0:
                                sent1 = prev_sent
                                sent2 = " ".join(word_list[1:])
                                connector = word
                                fo.write(
                                    external_punc_remover(sent1)
                                    + "\t"
                                    + external_punc_remover(sent2)
                                    + "\t"
                                    + connector
                                    + "\n"
                                )

                            # pass # Here is where it would be previous sentence
                    else:
                        continue

                if word.lower() == "and" or word.lower() == "but":
                    if rel_list[index] == "cc":
                        parent = parent_list[index] - 1
                        if has_subj(rel_list, parent_list, parent):
                            has_conjunct = 0
                            for rel_index, rel in enumerate(rel_list):
                                if (
                                    rel == "conj"
                                    and parent_list[rel_index] - 1 == parent
                                    and has_subj(rel_list, parent_list, rel_index)
                                    and rel_index > index
                                ):
                                    sent1 = get_all_descendants(
                                        word_list, parent_list, parent, rel_list, index, index
                                    )
                                    sent2 = get_all_descendants(
                                        word_list, parent_list, rel_index, rel_list, index, index
                                    )
                                    connector = word
                                    fo.write(
                                        external_punc_remover(sent1)
                                        + "\t"
                                        + external_punc_remover(sent2)
                                        + "\t"
                                        + connector
                                        + "\n"
                                    )
                                    has_conjunct = 1
                            if not has_conjunct:
                                if index == 0:
                                    sent1 = prev_sent
                                    sent2 = " ".join(word_list[1:])
                                    connector = word
                                    fo.write(
                                        external_punc_remover(sent1)
                                        + "\t"
                                        + external_punc_remover(sent2)
                                        + "\t"
                                        + connector
                                        + "\n"
                                    )

                                # pass # Here is where it would be previous sentence

        prev_sent = " ".join(word_list)
        word_list = []
        parent_list = []
        rel_list = []
        root_index = -1

        marked = 0
        continue

    this_rel = line.split("(")[0]
    end_parts = line[line.index("(") + 1 :].split()
    this_word = end_parts[1].split("-")[0]
    this_parent = int(end_parts[0].split("-")[-1][:-1])

    word_list.append(this_word)
    rel_list.append(this_rel)
    parent_list.append(this_parent)

    if this_word.lower() == "and" or this_word.lower() == "but":
        if this_rel == "cc":

            marked = 1

    if (
        this_word.lower() == "because"
        or this_word.lower() == "if"
        or this_word.lower() == "before"
        or this_word.lower() == "though"
        or this_word.lower() == "so"
    ):

        if this_rel == "mark" or this_rel == "advmod":
            marked = 1

    if this_word.lower() == "when":
        if this_rel == "advmod":
            marked = 1
