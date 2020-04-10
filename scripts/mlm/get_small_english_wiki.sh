wiki_path=$1

mkdir -p $wiki_path/wikipedia_corpus_small
head -3978309 $wiki_path/train_en.txt > $wiki_path/wikipedia_corpus_small/train.txt
head -10001 $wiki_path/test_en.txt > $wiki_path/wikipedia_corpus_small/test.txt
tail -8438 $wiki_path/train_en.txt > $wiki_path/wikipedia_corpus_small/valid.txt
