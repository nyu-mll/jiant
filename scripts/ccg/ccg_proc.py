import os
import sys

# First create the training set
fo = open("ccg.train", "w")

# Standard training split
directories = [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
]

for directory in directories:
    for filename in os.listdir(directory):
        fi = open(directory + "/" + filename)

        # Extract sentences and just their tags
        for line in fi:
            if "<" in line:
                sentence = []
                tags = []

                items = line.strip().split("<")
                for item in items:
                    parts = item.split()

                    if parts[0] == "L":
                        sentence.append(parts[4])
                        tags.append(parts[1])

                # Write to file
                if len(sentence) == len(tags):
                    fo.write(" ".join(sentence) + "\t" + " ".join(tags) + "\n")

# Then the dev set
fo = open("ccg.dev", "w")

# Standard dev split
directories = ["19", "20", "21"]

for directory in directories:
    for filename in os.listdir(directory):
        fi = open(directory + "/" + filename)

        # Extract sentences and just their tags
        for line in fi:
            if "<" in line:
                sentence = []
                tags = []

                items = line.strip().split("<")
                for item in items:
                    parts = item.split()

                    if parts[0] == "L":
                        sentence.append(parts[4])
                        tags.append(parts[1])

                # Write to file
                if len(sentence) == len(tags):
                    fo.write(" ".join(sentence) + "\t" + " ".join(tags) + "\n")


# And finally the test set
fo = open("ccg.test", "w")

# Standard test split
directories = ["22", "23", "24"]

for directory in directories:
    for filename in os.listdir(directory):
        fi = open(directory + "/" + filename)

        # Extract sentences and just their tags
        for line in fi:
            if "<" in line:
                sentence = []
                tags = []

                items = line.strip().split("<")
                for item in items:
                    parts = item.split()

                    if parts[0] == "L":
                        sentence.append(parts[4])
                        tags.append(parts[1])

                # Write to file
                if len(sentence) == len(tags):
                    fo.write(" ".join(sentence) + "\t" + " ".join(tags) + "\n")
