import sys

from nltk.tokenize.moses import MosesTokenizer

TOK = MosesTokenizer()

fi = open(sys.argv[1], "r")
fo = open(sys.argv[1] + ".moses", "w")

for line in fi:
    parts = line.strip().split("\t")
    old_toks = parts[0].split()
    new_toks = TOK.tokenize(parts[0])
    tags = parts[1].split()

    new_tags = []
    tag_counter = 0
    next_covered = 0
    for index, word in enumerate(new_toks):
        if next_covered > 0:
            next_covered -= 1
            continue

        if word == old_toks[tag_counter].replace("&", "&amp;").replace("'", "&apos;"):
            new_tags.append(tags[tag_counter])
            tag_counter += 1
        else:
            for i in range(7):
                if word + "".join(new_toks[index + 1 : index + 1 + i + 1]) == old_toks[
                    tag_counter
                ].replace("&", "&amp;").replace("'", "&apos;"):
                    for k in range(i + 2):
                        new_tags.append(tags[tag_counter])
                    tag_counter += 1
                    next_covered = i + 1
                    break

            if next_covered == 0:
                print("ERROR!", word, word + new_toks[index + 1], old_toks[tag_counter])
                print(" ".join(old_toks))
                print(" ".join(new_toks))
                print(" ")
    if len(new_tags) != len(new_toks):
        print("MISMATCH!!!")

    fo.write(
        (" ".join(new_toks) + "\t" + " ".join(new_tags) + "\n")
        .replace("&amp;", "&")
        .replace("&apos;", "'")
    )
