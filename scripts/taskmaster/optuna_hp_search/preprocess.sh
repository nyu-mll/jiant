
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/spr1/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/spr2/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/dpr/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/dep_ewt/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/ontonotes/const/pos/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/ontonotes/const/nonterminal/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/ontonotes/srl/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/ontonotes/ner/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/ontonotes/coref/*.json
python probing/retokenize_edge_data.py -t albert-xxlarge-v2 {os.path.join(DATA_DIR, edges/semeval/*.json
python scripts/ccg/align_tags_to_bert.py -t albert-xxlarge-v2 -d {os.path.join(DATA_DIR, ccg