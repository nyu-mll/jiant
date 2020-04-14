PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/spr1/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/spr2/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/dpr/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/dep_ewt/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/ontonotes/const/pos/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/ontonotes/const/nonterminal/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/ontonotes/srl/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/ontonotes/ner/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/ontonotes/coref/*.json" sbatch ~/cpu.sbatch
PROG="probing/retokenize_edge_data" ARGS="-t roberta-large edges/semeval/*.json" sbatch ~/cpu.sbatch
PROG="scripts/ccg/align_tags_to_bert" ARGS="-t roberta-large -d ccg" sbatch ~/cpu.sbatch
