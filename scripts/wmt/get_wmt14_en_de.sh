#########################
# Download data / tools #
#########################

git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git

wget http://statmt.org/wmt13/training-parallel-europarl-v7.tgz
wget http://statmt.org/wmt13/training-parallel-commoncrawl.tgz
wget http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz
wget http://statmt.org/wmt14/test-full.tgz

tar xvzf training-parallel-europarl-v7.tgz
tar xvzf training-parallel-commoncrawl.tgz
tar xvzf training-parallel-nc-v12.tgz
tar xvzf test-full.tgz

########################
# Source Preprocessing #
########################

cat training/europarl-v7.de-en.en commoncrawl.de-en.en training/news-commentary-v12.de-en.en \
    | perl mosesdecoder/scripts/tokenizer/normalize-punctuation.perl en \
    | perl mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl > train_valid.tok.en

awk 'NR % 100 != 0 {print $0 > "train.tmp.en"} NR % 100 == 0 {print $0 > "valid.tmp.en"}' train_valid.tok.en

sed -n 's/<seg id=\"[0-9]*\">\(.*\)<\/seg>/\1/p' test-full/newstest2014-deen-src.en.sgm > test.en

########################
# Target Preprocessing #
########################

cat training/europarl-v7.de-en.de commoncrawl.de-en.de training/news-commentary-v12.de-en.de \
    | perl mosesdecoder/scripts/tokenizer/normalize-punctuation.perl de \
    | perl mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl \
    | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l de > train_valid.tok.de

awk 'NR % 100 != 0 {print $0 > "train.tok.de"} NR % 100 == 0 {print $0 > "valid.tok.de"}' train_valid.tok.de

sed -n 's/<seg id=\"[0-9]*\">\(.*\)<\/seg>/\1/p' test-full/newstest2014-deen-ref.de.sgm \
    | perl mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l de > test.tok.de

### Apply BPE on target side only
python subword-nmt/learn_bpe.py -s 40000 < train.tok.de > bpe.de.code
python subword-nmt/apply_bpe.py -c bpe.de.code < train.tok.de > train.tmp.de
python subword-nmt/apply_bpe.py -c bpe.de.code < valid.tok.de > valid.tmp.de
python subword-nmt/apply_bpe.py -c bpe.de.code < test.tok.de > test.de

### Filter training/valid sets
perl mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 train.tmp en de train 1 200
perl mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 valid.tmp en de valid 1 200

paste train.en train.de > train.txt
paste valid.en valid.de > valid.txt
paste test.en train.de > test.txt
