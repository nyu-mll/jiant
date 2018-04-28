# Script for downloading GLUE data, based off script included in SentEval
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and tokenize data with MOSES tokenizer
#

data_path='glue_data'
mkdir ${data_path}

### Task links ###
# Single-sentence
COLA='https://nyu-mll.github.io/CoLA/cola_public.zip'
COLA_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FCoLA.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=Ym5KZTGEGW%2BWp2gBF3Et%2FXN%2Bjvmq%2BPh26nAdkAsUgh0vrVoi8tz0kdRI3Qb34OVtcQxc%2BytaANJkJd1iots6FPwjqmi4aMDXVP1LNusXgYqFnREy%2FtwcdY%2BWdGQkbtiGguILl0Us2w0G%2BzUgkaJK%2BNuriGzKpX0bRoI%2F6bay%2BgxH7QMvd41e2coqscDUnsZJE4KT4O1jf3c5A3fEfCcbHO%2B9h83RattV5bV6nU2wFWrfQkRSUE70ahNZ86srFaT6Mz150OMvlaOBUnDfBDlx7PpywgpTiP6RwsymIM%2BXzxCQ1T5%2FLAqmKqLqf8SkuXsIzbpNFe7PN1PMvh5Ac6czow%3D%3D'
SST='https://raw.githubusercontent.com/PrincetonML/SIF/master/data'
SST_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FSST-2.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuOO%2FOcfrwVQssOKbyikgsDy%2B%2BazhhbQvb0RboGCSGuhF0T%2BIOm2Xxca80SWqSEkXMkdiajJ43u9ARbr1X86ZqDF1jNP28LFOlTshHgUhnZqvkiyzQJ4FHfyOgb0nXIsdT1ChCGzzs3gcpM5wkD4ZoxoxjWVL4RvakWqpstX4n%2FO8vsLdGbNLpsf6W6OpSMvK9NCxYjlPmEHvxbl9swV1LYKRkZ9LYq%2FhebKt8mKcN5tokWAUufjZivgq7jTHB1mvhLfX%2BIeoxKgAsjuXqZHAHpdeSAqoPV53NcV7Jwab%2Fj%2F1ZsWleXW0S5s1rxcwJLDPPr9Lv2LaxpZNOplk5rImw%3D%3D'

# Paraphrase / similarity
MRPC='https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi'
MRPC_TEST='https://goo.gl/jB77No'
QQP='http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
QQP_TEST='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/tsvsWithoutLabels%2FQQP.tsv?alt=media&token=ed50f247-926c-4c4a-939b-f4f1d8a37f70'
STSBenchmark='http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'
STSBenchmark_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FSTS-B.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=ecO1KXSD44U5SkDEAQQq%2FaFO4IjaTFpP4lZQ4QbysVU%2ByPjvFgreYRrJxDgbr52q9369VQ5W98A%2Fy%2Bs6kCQZv4XlmLqbTw9Xs4OorvTq1WuzsdNeSMdicEKrefF4u9KiUJ6SpkHaF%2FFA16Y3wGQEyvINVi8BgyOxCijN29Ft%2F%2FjGlXkQ4UmXOg8GqnieiskzuAKbAZN3yWlvSL8OG7CZx1ft7obBMwxq5C4co55iwUnzQaQoCXxU6jyI%2FQGl8R3qhGzkjHMlRTPLJHu0kJAxq7fft%2FUDW8lYhQd%2FBMAlzBos54yP6xBVJSBKgft5Z%2BThpIi6KRabKnP4lqQluj%2BudQ%3D%3D'

# NLI
MNLI='https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'
MNLI_MATCHED_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FMNLI-m.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=dSW7k98DiOFYoASdg2yumf8139lOKXwRjXoMQY9wIeQiWMWoOIDs8iTwojC4ArMq30%2FGlUBqn1T7J47TbZCaM9mXIE7574brF%2F7HWOJlmYW21WEeuSyAxlPyXGqyDx5uFYIcJkwdDHfGEabTiIcpw1aYfa8jVbTonD80Ei5aGRrdI%2F9lDi2aYoB6rjNB5dE0gRKWqmNqmiIyVGwFZ%2B7Foe%2FcJJt2W7AVR%2B3OAE2hryzdTcFoLevvsFiCgoQb%2BhprEmZq3f5P0yF2BZQYWPFwe9tYNflpLx4R9M6mmpZ06wrPvJPdWO9R3yPiLAzagUJFJHWj36Bp5iZT62nBAVC%2FDA%3D%3D'
MNLI_MISMATCHED_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FMNLI-mm.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=AJvslVc0qt57qU47lgwI3QBYo2lTC87%2FCcQQU1913LdSl2P76HQRfpRK%2BN7%2Fxsc6q5P%2BefkQCdp13HgVvJXA5OQzZmWlBeV%2BAH5nKDAHD1LKdf5GX5pQUBdfD03r40lT8MQeKz9GVrXCc3kVq6NkaO3jw0Eq0T%2BnlnyfBlEBZcjU4YqOiAMX5x%2BfJqsljDHQF%2FO4tXtlr65pvQy2LAwk4tH6o6Oo03F8F1Q%2Bzf%2Br%2F1VqVCYRjvu3F1vHJm3knAojEycSFijonDk6FzGtFVohGQKnh2%2BG3OR9%2FDKFqYPcgdmtAW%2BGr381ybZTGVKTpsVaNMBDkLVqAuROPB%2B5vuhDeQ%3D%3D'
QNLI='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fqnli.zip?alt=media&token=9fc4c8a0-a8d0-438c-a200-34245ebebb7a'
QNLI_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FQNLI.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=EZfMkJ6ezAYPD0qLxHkaaKtofQwhSZ46707Q0Oz48Hy4LhQQZ5OXX%2FWJFE9M9FejFzyyRlG1ZItYKzUt%2BzllGZlahn7duIrPybet5KGPczGmxuUktuP6Ou1KOzrAAhrX5yyWaRGu0zCXfLAuRi%2BUTOGsBgCKgdTX8pTldzfv0HgTQ6Krht86IiXg9%2Buh6N2clTedvluLGz5fzU4oXfIouDVHU9oG1X2sivu%2FYGyo%2FAACwt4qtfdybd3nKyx%2FP8FfbjopyWMPMemGTH9qWT%2FocM6b7QJYM7uFrTwWcTgiQ1%2FCxY%2FXnEfvcRamHS5Lg%2Fu7vqgtd2gyrQF6YlMwG3FidQ%3D%3D'
RTE='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Frte.zip?alt=media&token=f6c12d50-092c-4b9b-b927-3504dbf16e48'
RTE_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FRTE.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=W0fENRE2entXz8bKTOLZADDu0QJrujSeyOUkoRL69yfGi5ihZCc3YyBB3jGV%2FsqGB465qrE8Jj2bpOBU93KSaxtazenOKGHy7GhdxUPXoIPDnvJ8GuqRvo2flGihBwcAUxPQyJVw8gtoLLtGToo%2FAW3c7dVLyIPDPoYLX4D4HcYSDejXUCvrt4LaRqrIGq%2F%2FbTWSEEpDRJ7XYs7fsEKcFbhIAVlVEDjY3B4543fpYlFf05dE9E4xShxwDycoZKA3etuH4PQO1F%2BkWPqnFl379crknfvmnEC9bvfpbA1c4iE2fZOjqEwezpTPeqEx5oLKEgOscAKcUnFLtGUvIegMwA%3D%3D'
SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
WNLI='https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fwnli.zip?alt=media&token=413281e0-1b1e-44df-a02b-5ea1aa067b1c'
WNLI_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FWNLI.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=GMlcOASCqXgHROhC3u7G3XktWrHa2ZKwenNkbUz7LMfMsXjxL2qz9RTS8BgtoJ7OwJOkkOeo0gqgebzBhWBQlN0UuAkOAZtMor9gzVe6P%2BCoUvIP6qbWRRkG8nBPguBe3Ff6%2BbCqY1ptV1dOns2elZOSh3G0ujcYB20kCLt5oVACDhbAhApFB%2FMe6iN80xTgEHHRMNzB%2FHsNNz31%2BsFa2kk%2F7xker%2B0l0kfMG%2FYndcRPoGKRb%2BzLQ73%2Fo6YDMf5E22c01CQAmF2ezddwycfPC%2FGE6j4cV2rbnepfHPdHoQJxo3ppjSAlKnJWGrgiOf7GIXPNUbe060h6iS4AHn%2F%2Bfg%3D%3D'
DIAGNOSTIC_TEST='https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D'

### CoLA ###
wget ${COLA} -O cola_public.zip
unzip cola_public.zip
rm -rf __MACOSX
mv public ${data_path}/CoLA
COLA_PATH=${data_path}/CoLA
mv cola_public.zip ${COLA_PATH}
cp ${COLA_PATH}/raw/in_domain_train.tsv ${COLA_PATH}/train.tsv
cat ${COLA_PATH}/raw/in_domain_dev.tsv > ${COLA_PATH}/dev.tsv
cat ${COLA_PATH}/raw/out_of_domain_dev.tsv >> ${COLA_PATH}/dev.tsv
wget ${COLA_TEST} -O ${COLA_PATH}/test.tsv 
rm -rf ${COLA_PATH}/.DS_Store
rm -rf ${COLA_PATH}/cola_public.zip

### SST ###
SST_PATH=${data_path}/SST
mkdir -p ${SST_PATH}
wget ${SST}/sentiment-train -O ${SST_PATH}/train.tsv
wget ${SST}/sentiment-dev -O ${SST_PATH}/dev.tsv
wget ${SST_TEST} -O ${SST_PATH}/test.tsv 

### MRPC ###
# This extraction needs "cabextract" to extract the MSI file
# sudo apt-get install cabextract
# sudo yum install cabextract
# sudo brew install cabextract
MRPC_PATH=${data_path}/MRPC
mkdir ${MRPC_PATH}
curl -Lo ${MRPC_PATH}/MSRParaphraseCorpus.msi ${MRPC}
cabextract ${MRPC_PATH}/MSRParaphraseCorpus.msi -d ${MRPC_PATH}
cat ${MRPC_PATH}/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > ${MRPC_PATH}/train.tsv
cat ${MRPC_PATH}/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > ${MRPC_PATH}/test.tsv
rm ${MRPC_PATH}/_*
rm ${MRPC_PATH}/MSRParaphraseCorpus.msi

### QQP ###
QQP_PATH=${data_path}/QQP
mkdir -p ${QQP_PATH}
wget ${QQP} -O ${QQP_PATH}/train_and_dev.zip
mv ${QQP_PATH}/train_and_dev.zip ${QQP_PATH}/train.tsv
rm ${QQP_PATH}/train_and_dev.zip
wget ${QQP_TEST} -O ${QQP_PATH}/test.tsv

### STSBenchmark ###
STS_PATH=${data_path}/STSBenchmark
curl -Lo $data_path/Stsbenchmark.tar.gz $STSBenchmark
tar -zxvf $data_path/Stsbenchmark.tar.gz -C $data_path
rm $data_path/Stsbenchmark.tar.gz
mv $data_path/stsbenchmark ${STS_PATH}
mv ${STS_PATH}/sts-dev.csv ${STS_PATH}/dev.tsv
mv ${STS_PATH}/sts-train.csv ${STS_PATH}/train.tsv
rm ${STS_PATH}/sts-test.csv
wget ${STSBenchmark_TEST} -O ${STS_PATH}/test.tsv

### MNLI ###
MNLI_PATH=${data_path}/MNLI
mkdir ${MNLI_PATH}
wget ${MNLI} -O ${MNLI_PATH}/train_and_dev.zip
unzip ${MNLI_PATH}/train_and_dev.zip -d ${MNLI_PATH}
mv ${MNLI_PATH}/multinli_1.0/* ${MNLI_PATH}
wget ${MNLI_MATCHED_TEST} -O ${MNLI_PATH}/test_matched.tsv
wget ${MNLI_MISMATCHED_TEST} -O ${MNLI_PATH}/test_mismatched.tsv
rm -rf ${MNLI_PATH}/__MACOSX
rm -rf ${MNLI_PATH}/train_and_dev.zip

### QNLI ###
QNLI_PATH=${data_path}/QNLI
mkdir ${QNLI_PATH}
wget ${QNLI} -O ${QNLI_PATH}/train_and_dev.zip
unzip ${QNLI_PATH}/train_and_dev.zip -d ${QNLI_PATH}
rm ${QNLI_PATH}/train_and_dev.zip
mv ${QNLI_PATH}/qnli_train.tsv ${QNLI_PATH}/train.tsv
mv ${QNLI_PATH}/qnli_dev.tsv ${QNLI_PATH}/dev.tsv
wget ${QNLI_TEST} -O ${QNLI_PATH}/test.tsv

### SNLI ###
SNLI_PATH=${data_path}/SNLI
mkdir ${SNLI_PATH}
curl -Lo ${SNLI_PATH}/snli_1.0.zip $SNLI
unzip ${SNLI_PATH}/snli_1.0.zip -d ${SNLI_PATH}
rm ${SNLI_PATH}/snli_1.0.zip
rm -r ${SNLI_PATH}/__MACOSX

### RTE ###
RTE_PATH=${data_path}/RTE
mkdir ${RTE_PATH}
wget ${RTE} -O  ${RTE_PATH}/train_and_dev.zip
unzip ${RTE_PATH}/train_and_dev.zip -d ${RTE_PATH}
mv ${RTE_PATH}/rte_train.tsv ${RTE_PATH}/train.tsv
mv ${RTE_PATH}/rte_dev.tsv ${RTE_PATH}/dev.tsv
wget ${RTE_TEST} -O ${RTE_PATH}/test.tsv
rm ${RTE_PATH}/train_and_dev.zip

### WNLI ###
WNLI_PATH=${data_path}/WNLI
mkdir ${WNLI_PATH}
wget ${WNLI} -O ${WNLI_PATH}/train_and_dev.zip
unzip ${WNLI_PATH}/train_and_dev.zip -d ${WNLI_PATH}
mv ${WNLI_PATH}/wnli_train.tsv ${WNLI_PATH}/train.tsv
mv ${WNLI_PATH}/wnli_dev.tsv ${WNLI_PATH}/dev.tsv
wget ${WNLI_TEST} -O ${WNLI_PATH}/test.tsv
rm ${WNLI_PATH}/train_and_dev.zip

### Diagnostic ###
DIAGNOSTIC_PATH=${data_path}/diagnostic
mkdir ${DIAGNOSTIC_PATH}
wget ${DIAGNOSTIC_TEST} -O ${DIAGNOSTIC_PATH}/test.tsv
