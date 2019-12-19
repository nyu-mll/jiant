import sys

out_file = open(sys.argv[1], "w")
for l1, l2 in zip(open(sys.argv[2]), open(sys.argv[3])):
    out_file.write(l1.strip() + "\t" + l2)
out_file.close()
