fi1 = open("ccg.train", "r")
fi2 = open("ccg.test", "r")
fi3 = open("ccg.dev", "r")

fo1 = open("ccg_num.train", "w")
fo2 = open("ccg_num.test", "w")
fo3 = open("ccg_num.dev", "w")


tag2num = {}

counter = 0

for line in fi1:
    parts = line.strip().split("\t")
    tags = parts[1].split()

    for tag in tags:
        if tag not in tag2num:
            tag2num[tag] = str(counter)
            counter += 1

for line in fi2:
    parts = line.strip().split("\t")
    tags = parts[1].split()

    for tag in tags:
        if tag not in tag2num:
            tag2num[tag] = str(counter)
            counter += 1


for line in fi3:
    parts = line.strip().split("\t")
    tags = parts[1].split()

    for tag in tags:
        if tag not in tag2num:
            tag2num[tag] = str(counter)
            counter += 1

fi1.close()
fi2.close()
fi3.close()

print(counter)


fi1 = open("ccg.train", "r")
fi2 = open("ccg.test", "r")
fi3 = open("ccg.dev", "r")

for line in fi1:
    parts = line.strip().split("\t")
    sent = parts[0]
    tags = parts[1].split()

    nums = []
    for tag in tags:
        nums.append(tag2num[tag])

    fo1.write(sent + "\t" + " ".join(nums) + "\n")

for line in fi2:
    parts = line.strip().split("\t")
    sent = parts[0]
    tags = parts[1].split()

    nums = []
    for tag in tags:
        nums.append(tag2num[tag])

    fo2.write(sent + "\t" + " ".join(nums) + "\n")

for line in fi3:
    parts = line.strip().split("\t")
    sent = parts[0]
    tags = parts[1].split()

    nums = []
    for tag in tags:
        nums.append(tag2num[tag])

    fo3.write(sent + "\t" + " ".join(nums) + "\n")
