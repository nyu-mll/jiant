import os
import json

def preprocess():
    dataset_names = ['coordinating-conjunctions', 'whwords', 'definiteness', 'eos']

    for name in dataset_names:
        with open("ACCEPTABILITY/acceptability-{}_data.json".format(name)) as f:
            json_data = json.loads(f.read())

        with open("ACCEPTABILITY/acceptability-{}_metadata.json".format(name)) as f:
            json_metadata = json.loads(f.read())

        for d, m in zip(json_data, json_metadata):
            assert d['pair-id'] == m['pair-id'] 
            assert m['pair-id'] == m['corpus-sent-id']
    
        for i in range(1,11):
            split_d = {'train': [], 'dev': [], 'test': []}
            for d, m in zip(json_data, json_metadata):
                pair_id = d['pair-id']
                split = m['misc']["fold{}".format(i)]
                if name == 'eos':
                    split_d[split].append("{}\t{}\t{}\t{}\n".format(pair_id, d['context'], d['hypothesis'], d['label']))
                else:
                    split_d[split].append("{}\t{}\t{}\n".format(pair_id, d['context'], d['label']))
                
            save_path = "{}/fold{}/".format(name, i)
            if not os.path.exists(save_path): os.mkdirs(save_path)
            for split, data in split_d.items():
                with open(os.path.join(save_path, "{}.tsv".format(split)), 'w') as f:
                    f.writelines(data)

if __name__ == '__main__':
    preprocess() 
