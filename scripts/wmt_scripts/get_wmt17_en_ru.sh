# We download data and tools
wget http://data.statmt.org/wmt17/translation-task/preprocessed/ru-en/corpus.tc.en.gz
wget http://data.statmt.org/wmt17/translation-task/preprocessed/ru-en/corpus.tc.ru.gz

gunzip corpus.tc.en.gz
gunzip corpus.tc.ru.gz

git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git

perl mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 corpus.tc en ru clean 10 64

# We detokenize and detruecase to be consistent with other preprocessing
head -n 3200000 clean.en | perl mosesdecoder/scripts/tokenizer/detokenizer.perl | perl mosesdecoder/scripts/recaser/detruecase.perl > train.en

# We apply bpe on the target side only
head -n 3200000 clean.ru | python subword-nmt/learn_bpe.py -s 40000 > bpe_model
head -n 3200000 clean.ru | python subword-nmt/apply_bpe.py -c bpe_model > train.ru

# Merge source and target, shuffle and split in train / valid / test
paste train.en train.ru | shuf > all.txt
head -n 10000 all.txt > test.txt
head -n 20000 all.txt | tail -n 10000 > valid.txt
tail -n +20001 all.txt > train.txt

