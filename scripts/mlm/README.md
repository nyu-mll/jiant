# Downloading Wikipedia Corpus
We use a modified version of the data generation code from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data.

Run `bash create_wiki_data.sh $lang $save_directory`

For example, to download and process English Wikipedia and save it in `~/Download` directory, run
`bash create_wiki_data.sh en ~/Download`

The above command will download the entire English Wikipedia.

In our experiments, we only use a small subset (around 5% of) the entire English Wikipedia, which has the same number of sentences as Wikitext103. 
To get this subset, run `bash get_small_english_wiki.sh $path_to_wikicorpus_en`. where $path_to_wikicorpus_en is the directory where you saved the full processed `wikicorpus_en` corpus.
