import sys

fi1 = open(sys.argv[1], "r")
fi2 = open(sys.argv[2], "r")

fo1 = open(sys.argv[1] + ".zipped", "w")


lines1 = fi1.readlines()
lines2 = fi2.readlines()

for index, line in enumerate(lines1):
    fo1.write(line.strip().split("\t")[0] + "\t" + lines2[index].strip().split("\t")[1] + "\n")
