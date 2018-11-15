import sys

# Converts CoNLL format into the JSON format needed for edge probing
fi = open(sys.argv[1], "r")
fo = open(sys.argv[1].split(".")[0] + ".json", "w")

prev_line = "FILLER"
word_lines = []
examples = []

for line in fi:
	example_good = 1
	if len(prev_line) < 3:
		spans = []
		words = []
		
		for word_line in word_lines:
			#print(word_line)
			parts = word_line.split('\t')
			words.append(parts[1].replace('"', '\\"'))
			if "." not in parts[0]:
				this_id = int(parts[0])
				this_head = int(parts[6])
			else:
				example_good = 0
				this_id = 0
				this_head = 0
			if this_head == 0:
				this_head = this_id
			deprel = parts[7]
			#spans.append('{ "span1": [' + str(this_id - 1) + ',' + str(this_id) + '], "span2": [' + str(this_head - 1) + ',' + str(this_head) + '], "label": "' + deprel + '" }')
			spans.append('{"span1": [' + str(this_id - 1) + ', ' + str(this_id) + '], "span2": [' + str(this_head - 1) + ', ' + str(this_head) + '], "label": "' + deprel + '"}')


		#examples.append('{\n\t"text": "' + " ".join(words) + '",\n\t"targets": [\n\t' + "\n\t".join(spans) + '\n\t],\n\t"info": { "source": "UD_English-GUM" }\n}')		
		
		if example_good:
			examples.append('{"text": "' + " ".join(words) + '", "targets": [' + ", ".join(spans) + '], "info": {"source": "UD_English-EWT"}}')		



		word_lines = []
		
	
	elif line[0] != "#" and len(line.strip()) > 1:
		word_lines.append(line)
		
	prev_line = line.strip()


for example in examples:
	#fo.write(example + "\n\n")
	fo.write(example + "\n")
