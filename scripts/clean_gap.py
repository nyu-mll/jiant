
# i
import pandas as pd
import pickle

def getEnd(index, word):
    return index  + len(word)

def process_dataset(split):
    gap_text = pd.read_csv("/Users/yadapruksachatkun/jiant/data/gap-coreference/gap-"+split+".tsv",  header = 0, delimiter="\t")
    new_pandas = []
    for i in range(len(gap_text)):
        row = gap_text.iloc[i]
        text = row['Text']
        pronoun = row['Pronoun']
        pronoun_index = row["Pronoun-offset"]
        end_index_prnn = getEnd(pronoun_index, pronoun)
        first_index = row["A-offset"]
        first_word = row["A"]
        end_index = getEnd(first_index, first_word)
        label = row["A-coref"]
        new_pandas.append([text, pronoun_index, end_index_prnn, first_index, end_index, label])
        second_index = row["B-offset"]
        second_word = row["B"]
        end_index_b = getEnd(second_index, second_word)
        label_b = row['B-coref']
        new_pandas.append([text, pronoun_index, end_index_prnn, second_index, end_index_b, label_b])
    result = pd.DataFrame(new_pandas, columns=["text", "prompt_start_index", "prompt_end_index", "candidate_start_index", "candidate_end_index", "label"])
    pickle.dump(result, open("/Users/yadapruksachatkun/jiant/data/processed/gap-coreference/__"+split+"__", "wb"))
    # for i in rang len(new_one)):

process_dataset("test")
process_dataset("development")
process_dataset("validation")
