# This is a helper bash script. Execute run_stuff.sh, not this.

while getopts 'ivkmn:r:d:w:S:s:tvh:l:L:o:T:E:b:H:p:ecgP:qB:V:M:D:CX:GI:N:y:K:W:F:fA:' flag; do
    case "${flag}" in
        P) JIANT_PROJECT_PREFIX="${OPTARG}" ;;
        d) JIANT_DATA_DIR=${OPTARGS} ;;
        n) EXP_NAME="${OPTARG}" ;;
        r) RUN_NAME="${OPTARG}" ;;
        S) SEED="${OPTARG}" ;;
        I) GPUID="${OPTARG}" ;;

        w) word_embs="${OPTARG}" ;;
        C) char_embs=1 ;;
        f) fastText=1 ;;
	    F) FASTTEXT_MODEL_FILE="${OPTARG}" ;;
        e) ELMO=1 ;;
        c) COVE=1 ;;

        q) no_tqdm=1 ;;
        #t) SHOULD_TRAIN=0 ;;
        k) RELOAD_TASKS=1 ;;
        i) RELOAD_INDEX=1 ;;
        v) RELOAD_VOCAB=1 ;;
        #m) LOAD_MODEL=1 ;;

        B) BPP_BASE="${OPTARG}" ;;
        V) VAL_INTERVAL="${OPTARG}" ;;
        X) MAX_VALS="${OPTARG}" ;;
        T) train_tasks="${OPTARG}" ;;
        E) eval_tasks="${OPTARG}" ;;
        H) n_layers_highway="${OPTARG}" ;;
        A) n_heads="${OPTARG}" ;;
        l) LR="${OPTARG}" ;;
        #s) min_lr="${OPTARG}" ;;
        L) N_LAYERS_ENC="${OPTARG}" ;;
        o) OPTIMIZER="${OPTARG}" ;;
        h) d_hid="${OPTARG}" ;;
        b) BATCH_SIZE="${OPTARG}" ;;
        s) sent_enc="${OPTARG}" ;;
        #E) PAIR_ENC="${OPTARG}" ;;
        D) dropout="${OPTARG}" ;;
        #C) CLASSIFIER="${OPTARG}" ;;
        #N) FORCE_LOAD_EPOCH="${OPTARG}" ;;
        y) LR_DECAY="${OPTARG}" ;;
        K) task_patience="${OPTARG}" ;;
        p) patience="${OPTARG}" ;;
        W) weighting_method="${OPTARG}" ;;
        #s) scaling_method="${OPTARG}" ;;
    esac
done

EXP_DIR="${JIANT_PROJECT_PREFIX}/${EXP_NAME}/"
RUN_DIR="${JIANT_PROJECT_PREFIX}/${EXP_NAME}/${RUN_NAME}"
LOG_FILE="log.log"
mkdir -p ${EXP_DIR}
mkdir -p ${RUN_DIR}

declare -a ALLEN_ARGS
ALLEN_ARGS+=( --cuda ${GPUID} )
ALLEN_ARGS+=( --random_seed ${SEED} )
ALLEN_ARGS+=( --no_tqdm ${no_tqdm} )
ALLEN_ARGS+=( --log_file ${LOG_FILE} )
ALLEN_ARGS+=( --data_dir ${JIANT_DATA_DIR} )
ALLEN_ARGS+=( --exp_dir ${EXP_DIR} )
ALLEN_ARGS+=( --run_dir ${RUN_DIR} )
ALLEN_ARGS+=( --train_tasks ${train_tasks} )
ALLEN_ARGS+=( --eval_tasks ${eval_tasks} )
ALLEN_ARGS+=( --classifier ${CLASSIFIER} )
ALLEN_ARGS+=( --classifier_hid_dim ${d_hid_cls} )
ALLEN_ARGS+=( --max_seq_len ${max_seq_len} )
ALLEN_ARGS+=( --max_word_v_size ${VOCAB_SIZE} )
ALLEN_ARGS+=( --word_embs ${word_embs} )
ALLEN_ARGS+=( --word_embs_file ${WORD_EMBS_FILE} )
ALLEN_ARGS+=( --fastText_model_file ${FASTTEXT_MODEL_FILE} )
ALLEN_ARGS+=( --fastText ${fastText} )
ALLEN_ARGS+=( --char_embs ${char_embs} )
ALLEN_ARGS+=( --elmo ${ELMO} )
ALLEN_ARGS+=( --deep_elmo ${deep_elmo} )
ALLEN_ARGS+=( --cove ${COVE} )
ALLEN_ARGS+=( --d_word ${d_word} )
ALLEN_ARGS+=( --d_hid ${d_hid} )
ALLEN_ARGS+=( --sent_enc ${sent_enc} )
ALLEN_ARGS+=( --bidirectional ${bidirectional} )
ALLEN_ARGS+=( --n_layers_enc ${N_LAYERS_ENC} )
ALLEN_ARGS+=( --pair_enc ${PAIR_ENC} )
ALLEN_ARGS+=( --n_layers_highway ${n_layers_highway} )
ALLEN_ARGS+=( --n_heads ${n_heads} )
ALLEN_ARGS+=( --batch_size ${BATCH_SIZE} )
ALLEN_ARGS+=( --bpp_base ${BPP_BASE} )
ALLEN_ARGS+=( --optimizer ${OPTIMIZER} )
ALLEN_ARGS+=( --lr ${LR} )
ALLEN_ARGS+=( --min_lr ${min_lr} )
ALLEN_ARGS+=( --lr_decay_factor ${LR_DECAY} )
ALLEN_ARGS+=( --task_patience ${task_patience} )
ALLEN_ARGS+=( --patience ${patience} )
ALLEN_ARGS+=( --weight_decay ${WEIGHT_DECAY} )
ALLEN_ARGS+=( --dropout ${dropout} )
ALLEN_ARGS+=( --val_interval ${VAL_INTERVAL} )
ALLEN_ARGS+=( --max_vals ${MAX_VALS} )
ALLEN_ARGS+=( --weighting_method ${weighting_method} )
ALLEN_ARGS+=( --scaling_method ${scaling_method} )
ALLEN_ARGS+=( --scheduler_threshold ${SCHED_THRESH} )
#ALLEN_ARGS+=( --load_model ${LOAD_MODEL} )
ALLEN_ARGS+=( --reload_tasks ${RELOAD_TASKS} )
ALLEN_ARGS+=( --reload_indexing ${RELOAD_INDEX} )
ALLEN_ARGS+=( --reload_vocab ${RELOAD_VOCAB} )
ALLEN_ARGS+=( --do_train ${DO_TRAIN} )
ALLEN_ARGS+=( --do_eval ${DO_EVAL} )
ALLEN_ARGS+=( --do_probe ${DO_PROBE} )
ALLEN_ARGS+=( --train_for_eval ${TRAIN_FOR_EVAL} )

ALLEN_CMD="python ./src/main.py ${ALLEN_ARGS[@]}"
eval ${ALLEN_CMD}
