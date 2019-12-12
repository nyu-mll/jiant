TM_TARGET_TASK_NAMES=(rte-superglue boolq commitbank copa multirc record wic winograd-coreference cosmosqa)
TM_PROBING_TASK_NAMES=(edges-ner-ontonotes edges-srl-ontonotes edges-coref-ontonotes edges-spr1 edges-spr2 edges-dpr edges-rel-semeval se-probing-word-content se-probing-tree-depth se-probing-top-constituents se-probing-bigram-shift se-probing-past-present se-probing-subj-number se-probing-obj-number se-probing-odd-man-out se-probing-coordination-inversion edges-pos-ontonotes edges-nonterminal-ontonotes edges-dep-ud-ewt se-probing-sentence-length acceptability-wh acceptability-def acceptability-conj acceptability-eos cola)
export TM_MIXING_TASK_NAMES=(edges-ner-ontonotes edges-srl-ontonotes edges-coref-ontonotes edges-spr1 edges-spr2 edges-dpr edges-rel-semeval se-probing-word-content se-probing-tree-depth se-probing-top-constituents se-probing-bigram-shift se-probing-past-present se-probing-subj-number se-probing-obj-number se-probing-odd-man-out se-probing-coordination-inversion edges-pos-ontonotes edges-nonterminal-ontonotes edges-dep-ud-ewt se-probing-sentence-length acceptability-wh acceptability-def acceptability-conj acceptability-eos cola)

function run_all_intermediate_to_target() {
    # Do hyerparameter tuning search for the parameters
    # Usage: hyperparameter_sweep <task> <batch_size> <random_seed>
    TM_TARGET_TASK_NAMES=(rte-superglue boolq commitbank copa multirc record wic winograd-coreference cosmosqa)
    for task in ${TM_TARGET_TASK_NAMES[@]}
    do
        echo "ez_run_intermediate_to_target_task 2 commonsenseqa $task /scratch/pmh330/jiant-outputs/roberta-large-run2"
    done

}

function run_all_intermediate_to_probing() {
    TM_PROBING_TASK_NAMES=(edges-ner-ontonotes edges-srl-ontonotes edges-coref-ontonotes edges-spr1 edges-spr2 edges-dpr edges-rel-semeval se-probing-word-content se-probing-tree-depth se-probing-top-constituents se-probing-bigram-shift se-probing-past-present se-probing-subj-number se-probing-obj-number se-probing-odd-man-out se-probing-coordination-inversion edges-pos-ontonotes edges-nonterminal-ontonotes edges-dep-ud-ewt se-probing-sentence-length acceptability-wh acceptability-def acceptability-conj acceptability-eos cola)
    for task in ${TM_PROBING_TASK_NAMES[@]}
    do
        echo "ez_run_intermediate_to_probing 2 commonsenseqa $task /scratch/pmh330/jiant-outputs/roberta-large-run2"
    done

}

function run_all_intermediate_to_mixing() {
    TM_MIXING_TASK_NAMES=(edges-ner-ontonotes edges-srl-ontonotes edges-coref-ontonotes edges-spr1 edges-spr2 edges-dpr edges-rel-semeval se-probing-word-content se-probing-tree-depth se-probing-top-constituents se-probing-bigram-shift se-probing-past-present se-probing-subj-number se-probing-obj-number se-probing-odd-man-out se-probing-coordination-inversion edges-pos-ontonotes edges-nonterminal-ontonotes edges-dep-ud-ewt se-probing-sentence-length acceptability-wh acceptability-def acceptability-conj acceptability-eos cola)
    for task in ${TM_MIXING_TASK_NAMES[@]}
    do
        echo "ez_run_intermediate_to_mixing 2 commonsenseqa $task /scratch/pmh330/jiant-outputs/roberta-large-run2"
    done

}

export JIANT_PROJECT_PREFIX="/scratch/pmh330/jiant-outputs/roberta-large-run1"

run_all_intermediate_to_target
#run_all_intermediate_to_probing
#run_all_intermediate_to_mixing
