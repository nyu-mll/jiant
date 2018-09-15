import sys

fi = open(sys.argv[1], "r")
fo = open(sys.argv[2], "w")

label_dict = {}


def remove_punc(sent):
    words = sent.split()

    new_words = []
    seen_word = 0
    for word in words:
        if not seen_word and len(word) == 1 and not word.isalnum():
            pass
        else:
            seen_word = 1
            new_words.append(word)

    words = new_words[::-1]
    new_new_words = []
    seen_word = 0
    for word in words:
        if not seen_word and len(word) == 1 and not word.isalnum():
            pass
        else:
            seen_word = 1
            new_new_words.append(word)
    return " ".join(new_new_words[::-1])


label_count = 0
for line in fi:
    parts = line.strip().split("\t")
    sent1 = parts[0]
    sent2 = parts[1]
    label = parts[2]

    sent1 = remove_punc(sent1)
    sent2 = remove_punc(sent2)

    sent1 = sent1[0].upper() + sent1[1:] + " ."
    sent2 = sent2[0].upper() + sent2[1:] + " ."

    if label.lower() not in label_dict:
        label_dict[label] = label_count
        label_count += 1

    label = str(label_dict[label.lower()])

    fo.write(sent1 + "\t" + sent2 + "\t" + label + "\n")
