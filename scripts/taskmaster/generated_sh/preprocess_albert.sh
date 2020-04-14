PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/spr1/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/spr2/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/dpr/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/dep_ewt/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/ontonotes/const/pos/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/ontonotes/const/nonterminal/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/ontonotes/srl/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/ontonotes/ner/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/ontonotes/coref/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t albert-xxlarge-v2 edges/semeval/*.json" sbatch ~/cpu.sbatch
PROG="scripts/ccg/align_tags_to_bert" ARGS="-t albert-xxlarge-v2 -d ccg" sbatch ~/cpu.sbatch
