"""Core model and functions for building it."""
import copy
import json
import logging as log
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.training.metrics import Average
from sklearn.metrics import mean_squared_error

from jiant.allennlp_mods.elmo_text_field_embedder import (
    ElmoTextFieldEmbedder,
    ElmoTokenEmbedderWrapper,
)

from jiant.modules.edge_probing import EdgeClassifierModule
from jiant.modules.simple_modules import (
    Pooler,
    Classifier,
    SingleClassifier,
    PairClassifier,
    NullPhraseLayer,
    TokenMultiProjectionEncoder,
)
from jiant.modules.attn_pair_encoder import AttnPairEncoder
from jiant.modules.sentence_encoder import SentenceEncoder
from jiant.modules.bilm_encoder import BiLMEncoder
from jiant.modules.bow_sentence_encoder import BoWSentEncoder
from jiant.modules.elmo_character_encoder import ElmoCharacterEncoder
from jiant.modules.onlstm_phrase_layer import ONLSTMPhraseLayer
from jiant.modules.prpn_phrase_layer import PRPNPhraseLayer
from jiant.modules.onlstm.ON_LSTM import ONLSTMStack
from jiant.modules.prpn.PRPN import PRPN
from jiant.modules.seq2seq_decoder import Seq2SeqDecoder
from jiant.modules.span_modules import SpanClassifierModule
from jiant.huggingface_transformers_interface import input_module_uses_transformers
from jiant.tasks.edge_probing import EdgeProbingTask
from jiant.tasks.lm import AutoregressiveLanguageModelingTask, MaskedLanguageModelingTask
from jiant.tasks.lm_parsing import LanguageModelingParsingTask
from jiant.tasks.qa import MultiRCTask, ReCoRDTask
from jiant.tasks.seq2seq import Seq2SeqTask
from jiant.tasks.tasks import (
    GLUEDiagnosticTask,
    MultipleChoiceTask,
    PairClassificationTask,
    PairOrdinalRegressionTask,
    PairRegressionTask,
    RegressionTask,
    SequenceGenerationTask,
    SingleClassificationTask,
    SpanClassificationTask,
    SpanPredictionTask,
    STSBTask,
    TaggingTask,
    WiCTask,
    MRPCTask,
    QQPTask,
    SentenceOrderTask,
)
from jiant.utils import config
from jiant.utils.utils import (
    assert_for_log,
    get_batch_size,
    get_batch_utilization,
    get_elmo_mixing_weights,
    maybe_make_dir,
    format_output,
    uses_cuda,
)
from jiant.utils.data_loaders import get_tokenizer

# Elmo stuff
# Look in $ELMO_SRC_DIR (e.g. /usr/share/jsalt/elmo) or download from web
ELMO_OPT_NAME = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHTS_NAME = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_SRC_DIR = (
    os.getenv("ELMO_SRC_DIR")
    or "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/"
)
ELMO_OPT_PATH = os.path.join(ELMO_SRC_DIR, ELMO_OPT_NAME)
ELMO_WEIGHTS_PATH = os.path.join(ELMO_SRC_DIR, ELMO_WEIGHTS_NAME)


def build_sent_encoder(args, vocab, d_emb, tasks, embedder, cove_layer):
    # Build single sentence encoder: the main component of interest
    # Need special handling for language modeling
    # Note: sent_enc is expected to apply dropout to its input _and_ output if
    # needed.
    rnn_params = Params(
        {
            "input_size": d_emb,
            "bidirectional": True,
            "hidden_size": args.d_hid,
            "num_layers": args.n_layers_enc,
        }
    )
    if args.sent_enc == "onlstm":
        onlayer = ONLSTMPhraseLayer(
            vocab,
            args.d_word,
            args.d_hid,
            args.n_layers_enc,
            args.onlstm_chunk_size,
            args.onlstm_dropconnect,
            args.onlstm_dropouti,
            args.dropout,
            args.onlstm_dropouth,
            embedder,
            args.batch_size,
        )
        # The 'onlayer' acts as a phrase layer module for the larger SentenceEncoder module.
        sent_encoder = SentenceEncoder(
            vocab,
            embedder,
            args.n_layers_highway,
            onlayer.onlayer,
            skip_embs=args.skip_embs,
            dropout=args.dropout,
            sep_embs_for_skip=args.sep_embs_for_skip,
            cove_layer=cove_layer,
        )
        d_sent = args.d_word
        log.info("Using ON-LSTM sentence encoder!")
    elif args.sent_enc == "prpn":
        prpnlayer = PRPNPhraseLayer(
            vocab,
            args.d_word,
            args.d_hid,
            args.n_layers_enc,
            args.n_slots,
            args.n_lookback,
            args.resolution,
            args.dropout,
            args.idropout,
            args.rdropout,
            args.res,
            embedder,
            args.batch_size,
        )
        # The 'prpn' acts as a phrase layer module for the larger SentenceEncoder module.
        sent_encoder = SentenceEncoder(
            vocab,
            embedder,
            args.n_layers_highway,
            prpnlayer.prpnlayer,
            skip_embs=args.skip_embs,
            dropout=args.dropout,
            sep_embs_for_skip=args.sep_embs_for_skip,
            cove_layer=cove_layer,
        )
        d_sent = args.d_word
        log.info("Using PRPN sentence encoder!")
    elif (
        any(isinstance(task, AutoregressiveLanguageModelingTask) for task in tasks)
        or args.sent_enc == "bilm"
    ):
        assert_for_log(args.sent_enc in ["rnn", "bilm"], "Only RNNLM supported!")
        if any(isinstance(task, AutoregressiveLanguageModelingTask) for task in tasks):
            assert_for_log(
                not (
                    args.input_module == "elmo"
                    or args.input_module.startswith("bert")
                    or args.input_module.startswith("xlnet")
                ),
                f"Using input_module = {args.input_module} for language modeling is probably not a "
                "good idea, since it allows the language model to use information from the right-hand "
                "context.",
            )
        bilm = BiLMEncoder(d_emb, args.d_hid, args.d_hid, args.n_layers_enc)
        sent_encoder = SentenceEncoder(
            vocab,
            embedder,
            args.n_layers_highway,
            bilm,
            skip_embs=args.skip_embs,
            dropout=args.dropout,
            sep_embs_for_skip=args.sep_embs_for_skip,
            cove_layer=cove_layer,
        )
    elif args.sent_enc == "bow":
        sent_encoder = BoWSentEncoder(vocab, embedder)
        assert_for_log(
            not args.skip_embs, "Skip connection not currently supported with `bow` encoder."
        )
        d_sent = d_emb
    elif args.sent_enc == "rnn":
        sent_rnn = s2s_e.by_name("lstm").from_params(copy.deepcopy(rnn_params))
        sent_encoder = SentenceEncoder(
            vocab,
            embedder,
            args.n_layers_highway,
            sent_rnn,
            skip_embs=args.skip_embs,
            dropout=args.dropout,
            sep_embs_for_skip=args.sep_embs_for_skip,
            cove_layer=cove_layer,
        )
        d_sent = 2 * args.d_hid
    elif args.sent_enc == "none":
        # Expose word representation layer (GloVe, ELMo, etc.) directly.
        assert_for_log(
            args.skip_embs,
            "skip_embs is false and sent_enc is none, "
            "which means that your token representations are zero-dimensional. "
            "Consider setting skip_embs.",
        )
        phrase_layer = NullPhraseLayer(rnn_params["input_size"])
        sent_encoder = SentenceEncoder(
            vocab,
            embedder,
            args.n_layers_highway,
            phrase_layer,
            skip_embs=args.skip_embs,
            dropout=args.dropout,
            sep_embs_for_skip=args.sep_embs_for_skip,
            cove_layer=cove_layer,
        )
        d_sent = 0
    else:
        assert_for_log(
            False, f"Shared encoder layer specification `{args.sent_enc}` not recognized."
        )
    return sent_encoder, d_sent


def build_model(args, vocab, pretrained_embs, tasks, cuda_devices):
    """
    Build model according to args
    Returns: model which has attributes set in it with the attrbutes.
    """

    # Build embeddings.
    cove_layer = None
    if args.input_module.startswith("bert-"):
        from jiant.huggingface_transformers_interface.modules import BertEmbedderModule

        log.info(f"Using BERT model ({args.input_module}).")
        embedder = BertEmbedderModule(args)
        d_emb = embedder.get_output_dim()
    elif args.input_module.startswith("roberta-"):
        from jiant.huggingface_transformers_interface.modules import RobertaEmbedderModule

        log.info(f"Using RoBERTa model ({args.input_module}).")
        embedder = RobertaEmbedderModule(args)
        d_emb = embedder.get_output_dim()
    elif args.input_module.startswith("albert-"):
        from jiant.huggingface_transformers_interface.modules import AlbertEmbedderModule

        log.info(f"Using ALBERT model ({args.input_module}).")
        embedder = AlbertEmbedderModule(args)
        d_emb = embedder.get_output_dim()
    elif args.input_module.startswith("xlnet-"):
        from jiant.huggingface_transformers_interface.modules import XLNetEmbedderModule

        log.info(f"Using XLNet model ({args.input_module}).")
        embedder = XLNetEmbedderModule(args)
        d_emb = embedder.get_output_dim()
    elif args.input_module.startswith("openai-gpt"):
        from jiant.huggingface_transformers_interface.modules import OpenAIGPTEmbedderModule

        log.info(f"Using OpenAI GPT model ({args.input_module}).")
        embedder = OpenAIGPTEmbedderModule(args)
        d_emb = embedder.get_output_dim()
    elif args.input_module.startswith("gpt2"):
        from jiant.huggingface_transformers_interface.modules import GPT2EmbedderModule

        log.info(f"Using GPT-2 model ({args.input_module}).")
        embedder = GPT2EmbedderModule(args)
        d_emb = embedder.get_output_dim()
    elif args.input_module.startswith("transfo-xl-"):
        from jiant.huggingface_transformers_interface.modules import TransfoXLEmbedderModule

        log.info(f"Using Transformer-XL model ({args.input_module}).")
        embedder = TransfoXLEmbedderModule(args)
        d_emb = embedder.get_output_dim()
    elif args.input_module.startswith("xlm-"):
        from jiant.huggingface_transformers_interface.modules import XLMEmbedderModule

        log.info(f"Using XLM model ({args.input_module}).")
        embedder = XLMEmbedderModule(args)
        d_emb = embedder.get_output_dim()
    else:
        # Default case, used for ELMo, CoVe, word embeddings, etc.
        d_emb, embedder, cove_layer = build_embeddings(args, vocab, tasks, pretrained_embs)

    sent_encoder, d_sent_output = build_sent_encoder(
        args, vocab, d_emb, tasks, embedder, cove_layer
    )
    # d_task_input is the input dimension of the task-specific module
    # set skip_emb = 1 if you want to concatenate the encoder input with encoder output to pass
    # into task specific module.
    d_task_input = d_sent_output + (args.skip_embs * d_emb)

    # Build model and classifiers
    model = MultiTaskModel(args, sent_encoder, vocab, cuda_devices)
    build_task_modules(args, tasks, model, d_task_input, d_emb, embedder, vocab)
    model = model.cuda() if uses_cuda(cuda_devices) else model
    if isinstance(cuda_devices, list):
        model = nn.DataParallel(model, device_ids=cuda_devices)

    log.info("Model specification:")
    log.info(model)
    param_count = 0
    trainable_param_count = 0
    if args.list_params:
        log.info("Model parameters:")
    for name, param in model.named_parameters():
        param_count += np.prod(param.size())
        if param.requires_grad:
            trainable_param_count += np.prod(param.size())
            if args.list_params:
                log.info(
                    "\t%s: Trainable parameter, count %d with %s",
                    name,
                    np.prod(param.size()),
                    str(param.size()),
                )
        elif args.list_params:
            log.info(
                "\t%s: Non-trainable parameter, count %d with %s",
                name,
                np.prod(param.size()),
                str(param.size()),
            )
    log.info("Total number of parameters: {ct:d} ({ct:g})".format(ct=param_count))
    log.info("Number of trainable parameters: {ct:d} ({ct:g})".format(ct=trainable_param_count))
    return model


def build_embeddings(args, vocab, tasks, pretrained_embs=None):
    """ Build embeddings according to options in args """
    d_emb, d_char = 0, args.d_char

    token_embedders = {}
    # Word embeddings
    n_token_vocab = vocab.get_vocab_size("tokens")
    if args.input_module in ["glove", "fastText"] and pretrained_embs is not None:
        word_embs = pretrained_embs
        assert word_embs.size()[0] == n_token_vocab
        d_word = word_embs.size()[1]
        log.info("\tUsing pre-trained word embeddings: %s", str(word_embs.size()))
    elif args.input_module == "scratch":
        log.info("\tTraining word embeddings from scratch.")
        d_word = args.d_word
        word_embs = nn.Embedding(n_token_vocab, d_word).weight
    else:
        assert input_module_uses_transformers(args.input_module) or args.input_module in [
            "elmo",
            "elmo-chars-only",
        ], f"'{args.input_module}' is not a valid value for input_module."
        embeddings = None
        word_embs = None

    if word_embs is not None:
        embeddings = Embedding(
            num_embeddings=n_token_vocab,
            embedding_dim=d_word,
            weight=word_embs,
            trainable=(args.embeddings_train == 1),
            padding_index=vocab.get_token_index("@@PADDING@@"),
        )
        token_embedders["words"] = embeddings
        d_emb += d_word

    # Handle cove
    cove_layer = None
    if args.cove:
        assert embeddings is not None
        assert args.input_module == "glove", "CoVe requires GloVe embeddings."
        assert d_word == 300, "CoVe expects 300-dimensional GloVe embeddings."
        try:
            from jiant.modules.cove.cove import MTLSTM as cove_lstm

            # Have CoVe do an internal GloVe lookup, but don't add residual.
            # We'll do this manually in modules.py; see
            # SentenceEncoder.forward().
            cove_layer = cove_lstm(n_vocab=n_token_vocab, vectors=embeddings.weight.data)
            # Control whether CoVe is trainable.
            for param in cove_layer.parameters():
                param.requires_grad = bool(args.cove_fine_tune)
            d_emb += 600  # 300 x 2 for biLSTM activations
            log.info("\tUsing CoVe embeddings!")
        except ImportError as e:
            log.info("Failed to import CoVe!")
            raise e

    # Character embeddings
    if args.char_embs:
        log.info("\tUsing character embeddings!")
        char_embeddings = Embedding(vocab.get_vocab_size("chars"), d_char)
        filter_sizes = tuple([int(i) for i in args.char_filter_sizes.split(",")])
        char_encoder = CnnEncoder(
            d_char,
            num_filters=args.n_char_filters,
            ngram_filter_sizes=filter_sizes,
            output_dim=d_char,
        )
        char_embedder = TokenCharactersEncoder(
            char_embeddings, char_encoder, dropout=args.dropout_embs
        )
        d_emb += d_char
        token_embedders["chars"] = char_embedder
    else:
        log.info("\tNot using character embeddings!")

    # If we want separate ELMo scalar weights (a different ELMo representation for each classifier,
    # then we need count and reliably map each classifier to an index used by
    # allennlp internal ELMo.
    if args.sep_embs_for_skip:
        # Determine a deterministic list of classifier names to use for each
        # task.
        classifiers = sorted(set(map(lambda x: x._classifier_name, tasks)))
        # Reload existing classifier map, if it exists.
        classifier_save_path = args.run_dir + "/classifier_task_map.json"
        if os.path.isfile(classifier_save_path):
            loaded_classifiers = json.load(open(args.run_dir + "/classifier_task_map.json", "r"))
        else:
            # No file exists, so assuming we are just starting to pretrain. If pretrain is to be
            # skipped, then there's a way to bypass this assertion by explicitly allowing for
            # a missing classiifer task map.
            assert_for_log(
                args.do_pretrain or args.allow_missing_task_map,
                "Error: {} should already exist.".format(classifier_save_path),
            )
            if args.allow_missing_task_map:
                log.warning(
                    "Warning: classifier task map not found in model"
                    " directory. Creating a new one from scratch."
                )
            # default is always @pretrain@
            loaded_classifiers = {"@pretrain@": 0}
        # Add the new tasks and update map, keeping the internal ELMo index
        # consistent.
        max_number_classifiers = max(loaded_classifiers.values())
        offset = 1
        for classifier in classifiers:
            if classifier not in loaded_classifiers:
                loaded_classifiers[classifier] = max_number_classifiers + offset
                offset += 1
        log.info("Classifiers:{}".format(loaded_classifiers))
        open(classifier_save_path, "w+").write(json.dumps(loaded_classifiers))
        # Every index in classifiers needs to correspond to a valid ELMo output
        # representation.
        num_reps = 1 + max(loaded_classifiers.values())
    else:
        # All tasks share the same scalars.
        # Not used if input_module = elmo-chars-only (i.e. no elmo)
        loaded_classifiers = {"@pretrain@": 0}
        num_reps = 1
    if args.input_module.startswith("elmo"):
        log.info("Loading ELMo from files:")
        log.info("ELMO_OPT_PATH = %s", ELMO_OPT_PATH)
        if args.input_module == "elmo-chars-only":
            log.info("\tUsing ELMo character CNN only!")
            log.info("ELMO_WEIGHTS_PATH = %s", ELMO_WEIGHTS_PATH)
            elmo_embedder = ElmoCharacterEncoder(
                options_file=ELMO_OPT_PATH, weight_file=ELMO_WEIGHTS_PATH, requires_grad=False
            )
            d_emb += 512
        else:
            log.info("\tUsing full ELMo! (separate scalars/task)")
            if args.elmo_weight_file_path != "none":
                assert os.path.exists(args.elmo_weight_file_path), (
                    'ELMo weight file path "' + args.elmo_weight_file_path + '" does not exist.'
                )
                weight_file = args.elmo_weight_file_path
            else:
                weight_file = ELMO_WEIGHTS_PATH
            log.info("ELMO_WEIGHTS_PATH = %s", weight_file)
            elmo_embedder = ElmoTokenEmbedderWrapper(
                options_file=ELMO_OPT_PATH,
                weight_file=weight_file,
                num_output_representations=num_reps,
                # Dropout is added by the sentence encoder later.
                dropout=0.0,
            )
            d_emb += 1024

        token_embedders["elmo"] = elmo_embedder

    # Wrap ELMo and other embedders, and concatenates the resulting
    # representations alone the last (vector) dimension.
    embedder = ElmoTextFieldEmbedder(
        token_embedders,
        loaded_classifiers,
        elmo_chars_only=args.input_module == "elmo-chars-only",
        sep_embs_for_skip=args.sep_embs_for_skip,
    )

    assert d_emb, "You turned off all the embeddings, ya goof!"
    return d_emb, embedder, cove_layer


def build_task_modules(args, tasks, model, d_sent, d_emb, embedder, vocab):
    """
        This function gets the task-specific parameters and builds
        the task-specific modules.
    """

    # Attach task-specific params.
    for task in sorted(set(tasks), key=lambda x: x.name):
        task_params = get_task_specific_params(args, task.name)
        log.info(
            "\tTask '%s' params: %s",
            task.name,
            json.dumps(task_params.as_dict(quiet=True), indent=2),
        )
        # Store task-specific params in case we want to access later
        setattr(model, "%s_task_params" % task.name, task_params)

    # Actually construct modules.
    for task in sorted(set(tasks), key=lambda x: x.name):
        # If the name of the task is different than the classifier it should use
        # then skip the module creation.
        if task.name != model._get_task_params(task.name).get("use_classifier", task.name):
            log.info("Name of the task is different than the classifier it should use")
            continue
        build_task_specific_modules(task, model, d_sent, d_emb, vocab, embedder, args)


def build_task_specific_modules(task, model, d_sent, d_emb, vocab, embedder, args):
    """ Build task-specific components for a task and add them to model.
        These include decoders, linear layers for linear models.
    """
    task_params = model._get_task_params(task.name)
    if isinstance(task, SingleClassificationTask):
        module = build_single_sentence_module(
            task=task,
            d_inp=d_sent,
            project_before_pooling=model.project_before_pooling,
            params=task_params,
        )
        setattr(model, "%s_mdl" % task.name, module)
    elif isinstance(task, SentenceOrderTask):
        module = build_sop(task, d_sent, model, task_params, args)
        setattr(model, "%s_mdl" % task.name, module)
    elif isinstance(task, (PairClassificationTask, PairRegressionTask, PairOrdinalRegressionTask)):
        module = build_pair_sentence_module(task, d_sent, model=model, params=task_params)
        setattr(model, "%s_mdl" % task.name, module)
    elif isinstance(task, SpanPredictionTask):
        module = TokenMultiProjectionEncoder(
            projection_names=["span_start", "span_end"], d_inp=d_sent
        )
        setattr(model, "%s_mdl" % task.name, module)
    elif isinstance(task, LanguageModelingParsingTask):
        # The LM Parsing task does not support embeddings that use skip_embs.
        hid2voc = build_lm(task, d_sent, args)
        setattr(model, "%s_hid2voc" % task.name, hid2voc)
        setattr(model, "%s_mdl" % task.name, hid2voc)
    elif isinstance(task, MaskedLanguageModelingTask):
        module = build_mlm(model.sent_encoder._text_field_embedder)
        setattr(model, "%s_mdl" % task.name, module)
    elif isinstance(task, AutoregressiveLanguageModelingTask):
        assert not input_module_uses_transformers(args.input_module), (
            "our LM Task does not support transformers, if you need them, try to update",
            "corresponding parts of the code. You may find get_pretrained_lm_head and",
            "apply_lm_boundary_tokens from huggingface_transformers_interface.module useful,",
            "do check if they are working correctly though.",
        )
        d_sent = args.d_hid + (args.skip_embs * d_emb)
        hid2voc = build_lm(task, d_sent, args)
        setattr(model, "%s_hid2voc" % task.name, hid2voc)
    elif isinstance(task, SpanClassificationTask):
        module = build_span_classifier(task, d_sent, task_params)
        setattr(model, "%s_mdl" % task.name, module)
    elif isinstance(task, TaggingTask):
        hid2tag = build_tagger(task, d_sent, task.num_tags)
        setattr(model, "%s_mdl" % task.name, hid2tag)
    elif isinstance(task, MultipleChoiceTask):
        module = build_multiple_choice_module(
            task, d_sent, project_before_pooling=model.project_before_pooling, params=task_params
        )
        setattr(model, "%s_mdl" % task.name, module)
    elif isinstance(task, EdgeProbingTask):
        module = EdgeClassifierModule(task, d_sent, task_params)
        setattr(model, "%s_mdl" % task.name, module)
    elif isinstance(task, Seq2SeqTask):
        log.info("using {} attention".format(args.s2s["attention"]))
        decoder_params = Params(
            {
                "input_dim": d_sent,
                "target_embedding_dim": 300,
                "decoder_hidden_size": args.s2s["d_hid_dec"],
                "output_proj_input_dim": args.s2s["output_proj_input_dim"],
                "max_decoding_steps": args.max_seq_len,
                "target_namespace": task._label_namespace
                if hasattr(task, "_label_namespace")
                else "targets",
                "attention": args.s2s["attention"],
                "dropout": args.dropout,
                "scheduled_sampling_ratio": 0.0,
                "beam_size": args.s2s["beam_size"],
            }
        )
        decoder = Seq2SeqDecoder(vocab, **decoder_params)
        setattr(model, "%s_decoder" % task.name, decoder)
    elif isinstance(task, SequenceGenerationTask):
        decoder, hid2voc = build_decoder(task, d_sent, vocab, embedder, args)
        setattr(model, "%s_decoder" % task.name, decoder)
        setattr(model, "%s_hid2voc" % task.name, hid2voc)
    elif isinstance(task, (MultiRCTask, ReCoRDTask)):
        module = build_qa_module(task, d_sent, model.project_before_pooling, task_params)
        setattr(model, "%s_mdl" % task.name, module)
    else:
        raise ValueError("Module not found for %s" % task.name)


def get_task_specific_params(args, task_name):
    """ Search args for parameters specific to task.
    Args:
        args: main-program args, a config.Params object
        task_name: (string)
    Returns:
        AllenNLP Params object of task-specific params.
    """

    def _get_task_attr(attr_name, default=None):
        return config.get_task_attr(args, task_name, attr_name, default)

    # This is confusing because a lot of parameters get renamed.
    # TODO to replace with hierarchical configs and remove all the renaming and
    # boilerplate.
    params = {}
    params["cls_type"] = _get_task_attr("classifier")
    params["d_hid"] = _get_task_attr("classifier_hid_dim")
    params["pool_type"] = _get_task_attr("pool_type")
    params["d_proj"] = _get_task_attr("d_proj")
    params["shared_pair_attn"] = args.shared_pair_attn
    if args.shared_pair_attn:
        params["attn"] = args.pair_attn
        params["d_hid_attn"] = args.d_hid_attn
        params["dropout"] = args.classifier_dropout
    else:
        params["attn"] = _get_task_attr("pair_attn")
        params["d_hid_attn"] = _get_task_attr("d_hid_attn")
        params["dropout"] = _get_task_attr("classifier_dropout")

    # Used for span/edge classification. Other tasks can safely ignore.
    params["cls_loss_fn"] = _get_task_attr("span_classifier_loss_fn")
    params["cls_span_pooling"] = _get_task_attr("classifier_span_pooling")
    params["edgeprobe_cnn_context"] = _get_task_attr("edgeprobe_cnn_context")
    params["edgeprobe_symmetric"] = _get_task_attr("edgeprobe_symmetric")

    # For NLI probing tasks, might want to use a classifier trained on
    # something else (typically 'mnli').
    cls_task_name = _get_task_attr("use_classifier")
    # default to this task
    params["use_classifier"] = cls_task_name or task_name

    return Params(params)


def build_image_sent_module(task, d_inp, params):
    pooler = Pooler(project=True, d_inp=d_inp, d_proj=params["d_proj"])
    return pooler


def build_single_sentence_module(task, d_inp: int, project_before_pooling: bool, params: Params):
    """ Build a single sentence classifier

    args:
        - task (Task): task object, used to get the number of output classes
        - d_inp (int): input dimension to the module, needed for optional linear projection
        - project_before_pooling (bool): apply a projection layer before pooling.
        - params (Params): Params object with task-specific parameters

    returns:
        - SingleClassifier (nn.Module): single-sentence classifier consisting of
            (optional) a linear projection, pooling, and an MLP classifier
    """
    pooler = Pooler(
        project=project_before_pooling,
        d_inp=d_inp,
        d_proj=params["d_proj"],
        pool_type=params["pool_type"],
    )
    d_out = params["d_proj"] if project_before_pooling else d_inp
    classifier = Classifier.from_params(d_out, task.n_classes, params)
    module = SingleClassifier(pooler, classifier)
    return module


def build_sop(task, d_inp, model, params, args):
    project_before_pooling = True
    module = build_pair_sentence_module(task, d_inp, model, params, project_before_pooling)
    module.pooler.project = model.sent_encoder._text_field_embedder.model.pooler
    return module


def build_pair_sentence_module(task, d_inp, model, params, project_before_pooling=None):
    """ Build a pair classifier, shared if necessary """

    def build_pair_attn(d_in, d_hid_attn):
        """ Build the pair model """
        d_inp_model = 2 * d_in
        modeling_layer = s2s_e.by_name("lstm").from_params(
            Params(
                {
                    "input_size": d_inp_model,
                    "hidden_size": d_hid_attn,
                    "num_layers": 1,
                    "bidirectional": True,
                }
            )
        )
        pair_attn = AttnPairEncoder(model.vocab, modeling_layer, dropout=params["dropout"])
        return pair_attn

    # Build the "pooler", which does pools a variable length sequence
    #   possibly with a projection layer beforehand
    if params["attn"] and model.project_before_pooling:
        pooler = Pooler(project=False, d_inp=params["d_hid_attn"], d_proj=params["d_hid_attn"])
        d_out = params["d_hid_attn"] * 2
    else:
        if project_before_pooling is not None:
            model.project_before_pooling = project_before_pooling
        pooler = Pooler(
            project=model.project_before_pooling,
            d_inp=d_inp,
            d_proj=params["d_proj"],
            pool_type=params["pool_type"],
        )
        d_out = params["d_proj"] if model.project_before_pooling else d_inp

    # Build an attention module if necessary
    if params["shared_pair_attn"] and params["attn"]:  # shared attn
        if not hasattr(model, "pair_attn"):
            pair_attn = build_pair_attn(d_inp, params["d_hid_attn"])
            model.pair_attn = pair_attn
        else:
            pair_attn = model.pair_attn
    elif params["attn"]:  # non-shared attn
        pair_attn = build_pair_attn(d_inp, params["d_hid_attn"])
    else:  # no attn
        pair_attn = None

    # Build the classifier
    n_classes = task.n_classes if hasattr(task, "n_classes") else 1
    if model.uses_pair_embedding:
        # BERT/XLNet handle pair tasks by concatenating the inputs and classifying the joined
        # sequence, so we use a single sentence classifier
        if isinstance(task, WiCTask):
            d_out *= 3  # also pass the two contextual word representations
        classifier = Classifier.from_params(d_out, n_classes, params)
        module = SingleClassifier(pooler, classifier)
    else:
        d_out = d_out + d_inp if isinstance(task, WiCTask) else d_out
        classifier = Classifier.from_params(4 * d_out, n_classes, params)
        module = PairClassifier(pooler, classifier, pair_attn)
    return module


def build_lm(task, d_inp, args):
    """ Build LM components (just map hidden states to vocab logits) """
    hid2voc = nn.Linear(d_inp, args.max_word_v_size)
    return hid2voc


def build_mlm(embedder):
    " Build MLM components "
    lm_head = embedder.get_pretrained_lm_head()
    return lm_head


def build_span_classifier(task, d_sent, task_params):
    module = SpanClassifierModule(task, d_sent, task_params, num_spans=task.num_spans)
    return module


def build_tagger(task, d_inp, out_dim):
    """ Build tagger components. """
    hid2tag = nn.Linear(d_inp, out_dim)
    return hid2tag


def build_multiple_choice_module(task, d_sent, project_before_pooling, params):
    """ Basic parts for MC task: reduce a vector representation for each model into a scalar. """
    pooler = Pooler(
        project=project_before_pooling,
        d_inp=d_sent,
        d_proj=params["d_proj"],
        pool_type=params["pool_type"],
    )
    d_out = params["d_proj"] if project_before_pooling else d_sent
    choice2scalar = Classifier(d_out, n_classes=1, cls_type=params["cls_type"])
    return SingleClassifier(pooler, choice2scalar)


def build_decoder(task, d_inp, vocab, embedder, args):
    """ Build a task specific decoder """
    rnn = s2s_e.by_name("lstm").from_params(
        Params(
            {
                "input_size": embedder.get_output_dim(),
                "hidden_size": args.s2s["d_hid_dec"],
                "num_layers": args.s2s["n_layers_dec"],
                "bidirectional": False,
            }
        )
    )
    decoder = SentenceEncoder(vocab, embedder, 0, rnn)
    hid2voc = nn.Linear(args.s2s["d_hid_dec"], args.max_word_v_size)
    return decoder, hid2voc


def build_qa_module(task, d_inp, project_before_pooling, params):
    """ Build a simple QA module that
    1) pools representations (either of the joint (context, question, answer) or individually
    2) projects down to two logits
    3) classifier

    This module models each question-answer pair _individually_ """
    pooler = Pooler(
        project=project_before_pooling,
        d_inp=d_inp,
        d_proj=params["d_proj"],
        pool_type=params["pool_type"],
    )
    d_out = params["d_proj"] if project_before_pooling else d_inp
    classifier = Classifier.from_params(d_out, 2, params)
    return SingleClassifier(pooler, classifier)


class MultiTaskModel(nn.Module):
    """
    Giant model with task-specific components and a shared word and sentence encoder.
    This class samples the tasks passed in pretrained_tasks, and adds task specific components
    to the model.
    """

    def __init__(self, args, sent_encoder, vocab, cuda_devices):
        """ Args: sentence encoder """
        super(MultiTaskModel, self).__init__()
        self.sent_encoder = sent_encoder
        self._cuda_device = cuda_devices
        self.vocab = vocab
        self.utilization = Average() if args.track_batch_utilization else None
        self.elmo = args.input_module == "elmo"
        self.uses_pair_embedding = input_module_uses_pair_embedding(args.input_module)
        self.uses_mirrored_pair = input_module_uses_mirrored_pair(args.input_module)
        self.project_before_pooling = not (
            input_module_uses_transformers(args.input_module)
            and args.transfer_paradigm == "finetune"
        )  # Rough heuristic. TODO: Make this directly user-controllable.
        self.sep_embs_for_skip = args.sep_embs_for_skip

    def forward(self, task, batch, predict=False):
        """
        Pass inputs to correct forward pass
        Args:
            - task (tasks.Task): task for which batch is drawn
            - batch (Dict[str:Dict[str:Tensor]]): dictionary of (field, indexing) pairs,
                where indexing is a dict of the index namespace and the actual indices.
            - predict (Bool): passed to task specific forward(). If true, forward()
                should return predictions.
        Returns:
            - out: dictionary containing task outputs and loss if label was in batch
        """
        if self.utilization is not None:
            if "input1" in batch:
                self.utilization(get_batch_utilization(batch["input1"]))
            elif "input" in batch:
                self.utilization(get_batch_utilization(batch["input"]))
        if isinstance(task, SingleClassificationTask):
            out = self._single_sentence_forward(batch, task, predict)
        elif isinstance(task, GLUEDiagnosticTask):
            out = self._nli_diagnostic_forward(batch, task, predict)
        elif isinstance(
            task, (PairClassificationTask, PairRegressionTask, PairOrdinalRegressionTask)
        ):
            out = self._pair_sentence_forward(batch, task, predict)
        elif isinstance(task, MaskedLanguageModelingTask):
            out = self._masked_lm_forward(batch, task, predict)
        elif isinstance(task, AutoregressiveLanguageModelingTask):
            if isinstance(self.sent_encoder._phrase_layer, ONLSTMStack) or isinstance(
                self.sent_encoder._phrase_layer, PRPN
            ):
                out = self._lm_only_lr_forward(batch, task)
            else:
                out = self._lm_forward(batch, task, predict)
        elif isinstance(task, TaggingTask):
            out = self._tagger_forward(batch, task, predict)
        elif isinstance(task, MultipleChoiceTask):
            out = self._mc_forward(batch, task, predict)
        elif isinstance(task, EdgeProbingTask):
            # Just get embeddings and invoke task module.
            word_embs_in_context, sent_mask = self.sent_encoder(batch["input1"], task)
            module = getattr(self, "%s_mdl" % task.name)
            out = module.forward(
                batch=batch,
                word_embs_in_context=word_embs_in_context,
                sent_mask=sent_mask,
                task=task,
                predict=predict,
            )
        elif isinstance(task, SequenceGenerationTask):
            out = self._seq_gen_forward(batch, task, predict)
        elif isinstance(task, (MultiRCTask, ReCoRDTask)):
            out = self._multiple_choice_reading_comprehension_forward(batch, task, predict)
        elif isinstance(task, SpanClassificationTask):
            out = self._span_forward(batch, task, predict)
        elif isinstance(task, SpanPredictionTask):
            out = self._span_prediction_forward(batch, task, predict)
        else:
            raise ValueError("Task-specific components not found!")
        return out

    def _get_task_params(self, task_name):
        """ Get task-specific Params, as set in build_module(). """
        return getattr(self, "%s_task_params" % task_name)

    def _get_classifier(self, task):
        """ Get task-specific classifier, as set in build_module(). """
        # TODO: replace this logic with task._classifier_name?
        task_params = self._get_task_params(task.name)
        use_clf = task_params["use_classifier"]
        if use_clf in [None, "", "none"]:
            use_clf = task.name  # default if not set
        return getattr(self, "%s_mdl" % use_clf)

    def _single_sentence_forward(self, batch, task, predict):
        out = {}

        # embed the sentence
        word_embs_in_context, sent_mask = self.sent_encoder(batch["input1"], task)
        # pass to a task specific classifier
        classifier = self._get_classifier(task)
        logits = classifier(word_embs_in_context, sent_mask)
        out["logits"] = logits
        out["n_exs"] = get_batch_size(batch, self._cuda_device)
        if "labels" in batch:  # means we should compute loss
            if batch["labels"].dim() == 0:
                labels = batch["labels"].unsqueeze(0)
            elif batch["labels"].dim() == 1:
                labels = batch["labels"]
            else:
                labels = batch["labels"].squeeze(-1)
            out["loss"] = format_output(F.cross_entropy(logits, labels), self._cuda_device)
            out["labels"] = labels

        if predict:
            if isinstance(task, RegressionTask):
                if logits.ndimension() > 1:
                    assert (
                        logits.ndimension() == 2 and logits[-1] == 1
                    ), "Invalid regression prediction dimensions!"
                    logits = logits.squeeze(-1)
                out["preds"] = logits
            else:
                _, out["preds"] = logits.max(dim=1)
        return out

    def _nli_diagnostic_forward(self, batch, task, predict):
        out = {}

        # embed the sentence
        classifier = self._get_classifier(task)
        if self.uses_pair_embedding:
            sent, mask = self.sent_encoder(batch["inputs"], task)
            logits = classifier(sent, mask)
        else:
            sent1, mask1 = self.sent_encoder(batch["input1"], task)
            sent2, mask2 = self.sent_encoder(batch["input2"], task)
            logits = classifier(sent1, sent2, mask1, mask2)
        out["logits"] = logits
        out["n_exs"] = get_batch_size(batch, self._cuda_device)

        if "labels" in batch:
            if batch["labels"].dim() == 0:
                labels = batch["labels"].unsqueeze(0)
            elif batch["labels"].dim() == 1:
                labels = batch["labels"]
            else:
                labels = batch["labels"].squeeze(-1)
            out["loss"] = F.cross_entropy(logits, labels)

        if predict:
            _, predicted = logits.max(dim=1)
            out["preds"] = predicted

        return out

    def _span_forward(self, batch, task, predict):
        sent_embs, sent_mask = self.sent_encoder(batch["input1"], task)
        module = getattr(self, "%s_mdl" % task.name)
        out = module.forward(batch, sent_embs, sent_mask, task, predict, self._cuda_device)
        return out

    def _span_prediction_forward(self, batch, task, predict):
        sent_embs, sent_mask = self.sent_encoder(batch["inputs"], task)
        module = getattr(self, "%s_mdl" % task.name)
        logits_dict = module.forward(sent_embs, sent_mask)
        out = {
            "logits": logits_dict,
            "n_exs": get_batch_size(batch, self._cuda_device),
            "start_loss": F.cross_entropy(
                input=logits_dict["span_start"], target=batch["span_start"].long().squeeze(dim=1)
            ),
            "end_loss": F.cross_entropy(
                input=logits_dict["span_end"], target=batch["span_end"].long().squeeze(dim=1)
            ),
        }
        out["loss"] = (out["start_loss"] + out["end_loss"]) / 2

        # Form string predictions
        pred_span_start = torch.argmax(logits_dict["span_start"], dim=1)
        pred_span_end = torch.argmax(logits_dict["span_end"], dim=1)

        if predict:
            out["preds"] = {"span_start": pred_span_start, "span_end": pred_span_end}
        return out

    def _pair_sentence_forward(self, batch, task, predict):
        out = {}
        classifier = self._get_classifier(task)
        if isinstance(task, (MRPCTask, STSBTask, QQPTask)) and self.uses_mirrored_pair:
            # Mirrored pair is a trick used by GPT-like models in similarity tasks
            # TODO: Wic also falls into this type, although GPT paper didn't experiment
            #       with this task
            sent, mask = self.sent_encoder(batch["inputs"], task)
            sent_m, mask_m = self.sent_encoder(batch["inputs_m"], task)
            logits = classifier(sent, mask) + classifier(sent_m, mask_m)
        elif self.uses_pair_embedding:
            sent, mask = self.sent_encoder(batch["inputs"], task)
            # special case for WiC b/c we want to add representations of particular tokens
            if isinstance(task, WiCTask):
                logits = classifier(sent, mask, [batch["idx1"], batch["idx2"]])
            else:
                logits = classifier(sent, mask)
        else:
            sent1, mask1 = self.sent_encoder(batch["input1"], task)
            sent2, mask2 = self.sent_encoder(batch["input2"], task)
            if isinstance(task, WiCTask):
                logits = classifier(sent1, sent2, mask1, mask2, [batch["idx1"]], [batch["idx2"]])
            else:
                logits = classifier(sent1, sent2, mask1, mask2)
        out["n_exs"] = get_batch_size(batch, self._cuda_device)
        if "labels" in batch:
            labels = batch["labels"]
            labels = labels.squeeze(-1) if len(labels.size()) > 1 else labels
            if isinstance(task, RegressionTask):
                logits = logits.squeeze(-1) if len(logits.size()) > 1 else logits
                out["loss"] = F.mse_loss(logits, labels)
                labels = labels.detach()
                logits = logits.detach()
            else:
                out["loss"] = F.cross_entropy(logits, labels)
            out["labels"] = labels

        out["loss"] = format_output(out["loss"], self._cuda_device)
        out["logits"] = logits
        if predict:
            if isinstance(task, RegressionTask):
                if logits.ndimension() > 1:
                    assert (
                        logits.ndimension() == 2 and logits[-1] == 1
                    ), "Invalid regression prediction dimensions!"
                    logits = logits.squeeze(-1)
                out["preds"] = logits
            else:
                _, out["preds"] = logits.max(dim=1)
        return out

    def _seq_gen_forward(self, batch, task, predict):
        """ For sequence generation tasks """
        out = {}
        sent, sent_mask = self.sent_encoder(batch["inputs"], task)
        out["n_exs"] = get_batch_size(batch, self._cuda_device)

        decoder = getattr(self, "%s_decoder" % task.name)
        out.update(decoder.forward(sent, sent_mask, batch["targs"], generate=predict))
        # Loss is not computed during generation.
        if "loss" in out:
            task.scorer1(out["loss"].item())

        if "targs" in batch:
            # logits: batch_size * seq_len * tgt_voc_size
            target = batch["targs"]["words"][:, 1:].contiguous()
            target_mask = out["target_mask"]

            assert "predictions" in out
            out["logits"] = None
            out["labels"] = target

        return out

    def _tagger_forward(self, batch: dict, task: TaggingTask, predict: bool) -> dict:
        """
        This function is for sequence tagging (one-to-one mapping between words and tags).
        Args:
                batch: a dict of inputs and target tags
                task: TaggingTask
                predict: (boolean) predict mode (not supported)
        Returns
            out: (dict)
                - 'logits': output layer, dimension: [batchSize * task.max_seq_len, task.num_tags]
                - 'loss': size average CE loss
        """
        out = {}
        # batch[inputs] only has one item

        b_size, seq_len = list(batch["inputs"].values())[0].size()
        seq_len -= 2
        # Note: we are assuming there is one beginning and one ending token, when that no longer
        # holds, we need to refactor this by adjusting mask according to boundry function
        out["n_exs"] = get_batch_size(batch, self._cuda_device)
        sent, mask = self.sent_encoder(batch["inputs"], task)
        hid2tag = self._get_classifier(task)
        logits = hid2tag(sent[:, 1:-1, :]).view(b_size * seq_len, -1)
        targs = batch["targs"]["words"][:, :seq_len].contiguous().view(-1)
        if "mask" in batch:
            # Prevent backprop for tags generated for tokenization-introduced tokens
            # such as word boundaries
            batch_mask = batch["mask"][:, :seq_len]
            keep_idxs = torch.nonzero(batch_mask.contiguous().view(-1).data).squeeze()
            logits = logits.index_select(0, keep_idxs)
            targs = targs.index_select(0, keep_idxs)

            out["labels"] = targs
            out["logits"] = logits
        out["loss"] = format_output(F.cross_entropy(logits, targs), self._cuda_device)
        return out

    def _lm_forward(self, batch, task, predict):
        """Forward pass for LM model
        Args:
            batch: indexed input data
            task: (Task obejct)
            predict: (boolean) predict mode (not supported)
        return:
            out: (dict)
                - 'logits': output layer, dimension: [batchSize * timeSteps * 2, outputDim]
                            first half: [:batchSize*timeSteps, outputDim] is output layer from
                                forward layer
                            second half: [batchSize*timeSteps:, outputDim] is output layer from
                                backward layer
                - 'loss': size average CE loss
        """
        out = {}
        sent_encoder = self.sent_encoder
        assert_for_log(
            isinstance(sent_encoder._phrase_layer, BiLMEncoder),
            "Not using LM for language modeling task!",
        )
        assert_for_log(
            "targs" in batch and "words" in batch["targs"], "Batch missing target words!"
        )
        pad_idx = self.vocab.get_token_index(self.vocab._padding_token, "tokens")
        b_size, seq_len = batch["targs"]["words"].size()
        n_pad = batch["targs"]["words"].eq(pad_idx).sum().item()
        out["n_exs"] = format_output(((b_size * seq_len - n_pad) * 2), self._cuda_device)

        sent, mask = sent_encoder(batch["input"], task)
        sent = sent.masked_fill(1 - mask.byte(), 0)  # avoid NaNs

        # Split encoder outputs by direction
        split = int(self.sent_encoder._phrase_layer.get_output_dim() / 2)
        fwd, bwd = sent[:, :, :split], sent[:, :, split : split * 2]
        if split * 2 < sent.size(2):  # skip embeddings
            out_embs = sent[:, :, split * 2 :]
            fwd = torch.cat([fwd, out_embs], dim=2)
            bwd = torch.cat([bwd, out_embs], dim=2)

        # Forward and backward logits and targs
        hid2voc = getattr(self, "%s_hid2voc" % task.name)
        logits_fwd = hid2voc(fwd).view(b_size * seq_len, -1)
        logits_bwd = hid2voc(bwd).view(b_size * seq_len, -1)
        logits = torch.cat([logits_fwd, logits_bwd], dim=0)
        out["logits"] = logits
        trg_fwd = batch["targs"]["words"].view(-1)
        trg_bwd = batch["targs_b"]["words"].view(-1)
        targs = torch.cat([trg_fwd, trg_bwd], dim=0)
        assert logits.size(0) == targs.size(0), "Number of logits and targets differ!"
        out["loss"] = format_output(
            F.cross_entropy(logits, targs, ignore_index=pad_idx), self._cuda_device
        )
        task.scorer1(out["loss"].item())
        if predict:
            pass
        return out

    def _masked_lm_forward(self, batch, task, predict):
        """
        We currently only support RoBERTa-style dynamic masking, with the exact 
        setup and parameters as RoBERTa. 
        """
        out = {}
        tokenizer_name = self.sent_encoder._text_field_embedder.input_module
        text_embedder = self.sent_encoder._text_field_embedder
        vocab_size = text_embedder.model.embeddings.word_embeddings.num_embeddings
        input_key = text_embedder.tokenizer_required
        mask_idx = text_embedder._mask_id
        b_size, seq_len = batch["targs"].size()
        inputs = batch["input"][input_key]
        labels = batch["targs"]
        inputs, labels, _, _, _, _ = task.mlm_dynamic_masking(
            inputs, labels, mask_idx, tokenizer_name, self.sent_encoder
        )
        batch["input"][input_key] = inputs
        sent_embs, sent_mask = self.sent_encoder(batch["input"], task)
        module = getattr(self, "%s_mdl" % task.name)
        logits = module.forward(sent_embs)
        out["logits"] = logits
        out["loss"] = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        out["n_exs"] = format_output(b_size, self._cuda_device)
        return out

    def _mc_forward(self, batch, task, predict):
        """ Forward for a multiple choice question answering task """
        out = {}

        logits = []
        module = self._get_classifier(task)
        if self.uses_pair_embedding:
            for choice_idx in range(task.n_choices):
                sent, mask = self.sent_encoder(batch["choice%d" % choice_idx], task)
                logit = module(sent, mask)
                logits.append(logit)
        else:
            ctx, ctx_mask = self.sent_encoder(batch["question"], task)
            for choice_idx in range(task.n_choices):
                sent, mask = self.sent_encoder(batch["choice%d" % choice_idx], task)
                inp = torch.cat([ctx, sent], dim=1)
                inp_mask = torch.cat([ctx_mask, mask], dim=1)
                logit = module(inp, inp_mask)
                logits.append(logit)
        logits = torch.cat(logits, dim=1)
        out["logits"] = logits
        out["n_exs"] = get_batch_size(batch, self._cuda_device, keyword="choice0")
        if "label" in batch:
            labels = batch["label"]
            out["loss"] = format_output(F.cross_entropy(logits, labels), self._cuda_device)

        if predict:
            out["preds"] = logits.argmax(dim=-1)
        return out

    def _lm_only_lr_forward(self, batch, task):
        """Only left to right pass for LM model - non-bidirectional models.
           Used for language modeling training only in one direction.
        Args:
            batch: indexed input data
            task: (Task obejct)
        return:
            out: (dict)
                - 'logits': output layer, dimension: [batchSize * timeSteps, outputDim]
                    is output layer from forward layer
                - 'loss': size average CE loss
        """

        out = {}
        assert_for_log(
            "targs" in batch and "words" in batch["targs"], "Batch missing target words!"
        )
        pad_idx = self.vocab.get_token_index(self.vocab._padding_token, "tokens")
        b_size, seq_len = batch["targs"]["words"].size()
        # pad_idx is the token used to pad till max_seq_len
        n_pad = batch["targs"]["words"].eq(pad_idx).sum().item()
        # No of examples: only left to right, every unit in the sequence length is
        # a training example only once.
        out["n_exs"] = format_output(b_size * seq_len - n_pad, self._cuda_device)
        sent, mask = self.sent_encoder(batch["input"], task)
        sent = sent.masked_fill(1 - mask.byte(), 0)
        hid2voc = getattr(self, "%s_hid2voc" % task.name)
        logits = hid2voc(sent).view(b_size * seq_len, -1)
        out["logits"] = logits
        trg_fwd = batch["targs"]["words"].view(-1)
        assert logits.size(0) == trg_fwd.size(0), "Number of logits and targets differ!"
        out["loss"] = format_output(
            F.cross_entropy(logits, trg_fwd, ignore_index=pad_idx), self._cuda_device
        )
        task.scorer1(out["loss"].item())
        return out

    def _multiple_choice_reading_comprehension_forward(self, batch, task, predict):
        """ Forward call for multiple choice (selecting from a fixed set of answers)
        reading comprehension (have a supporting paragraph).

        Batch has a tensor of shape (n_questions, n_answers, n_tokens)
        """
        out = {}
        classifier = self._get_classifier(task)
        if self.uses_pair_embedding:
            # if using BERT/XLNet, we concatenate the passage, question, and answer
            inp = batch["psg_qst_ans"]
            ex_embs, ex_mask = self.sent_encoder(inp, task)
            logits = classifier(ex_embs, ex_mask)
            out["n_exs"] = get_batch_size(batch, self._cuda_device, keyword="psg_qst_ans")
        else:
            # else, we embed each independently and concat them
            psg_emb, psg_mask = self.sent_encoder(batch["psg"], task)
            qst_emb, qst_mask = self.sent_encoder(batch["qst"], task)

            if "ans" in batch:  # most QA tasks, e.g. MultiRC have explicit answer fields
                ans_emb, ans_mask = self.sent_encoder(batch["ans"], task)
                inp = torch.cat([psg_emb, qst_emb, ans_emb], dim=1)
                inp_mask = torch.cat([psg_mask, qst_mask, ans_mask], dim=1)
                out["n_exs"] = get_batch_size(batch, self._cuda_device, keyword="ans")
            else:  # ReCoRD inserts answer into the query
                inp = torch.cat([psg_emb, qst_emb], dim=1)
                inp_mask = torch.cat([psg_mask, qst_mask], dim=1)
                out["n_exs"] = get_batch_size(batch, self._cuda_device, keyword="qst")

            logits = classifier(inp, inp_mask)
        out["logits"] = logits
        if "label" in batch:
            out["loss"] = format_output(F.cross_entropy(logits, batch["label"]), self._cuda_device)

        if predict:
            if isinstance(task, ReCoRDTask):
                # For ReCoRD, we want the logits to make
                # predictions across answer choices
                # (which are spread across batches)
                out["preds"] = logits
            else:
                out["preds"] = logits.argmax(dim=-1)

        return out

    def get_elmo_mixing_weights(self, tasks=[]):
        """ Get elmo mixing weights from text_field_embedder. Gives warning when fails.
        args:
           - tasks (List[Task]): list of tasks that we want to get  ELMo scalars for.
        returns:
            - params Dict[str:float]: dictionary maybe layers to scalar params
        """
        params = {}
        if self.elmo:
            if not self.sep_embs_for_skip:
                tasks = [None]
            else:
                tasks = [None] + tasks
            for task in tasks:
                if task:
                    params[task._classifier_name] = get_elmo_mixing_weights(
                        self.sent_encoder._text_field_embedder, task=task
                    )
                else:
                    params["@pretrain@"] = get_elmo_mixing_weights(
                        self.sent_encoder._text_field_embedder, task=None
                    )
        return params


def input_module_uses_pair_embedding(input_module):
    """
    This function tells whether the input module concatenate the two sentences in a pair when
    running on pair tasks, like what GPT / BERT do on MNLI.
    It seems redundant now, but it allows us to load similar models from other sources later on
    """
    from jiant.huggingface_transformers_interface import input_module_uses_transformers

    return input_module_uses_transformers(input_module)


def input_module_uses_mirrored_pair(input_module):
    """
    This function tells whether the input model uses raw pair and mirrored pair simutaneously when
    running on symmetrical pair tasks, like what GPT do on STS-B
    """
    return (
        input_module.startswith("openai-gpt")
        or input_module.startswith("gpt2")
        or input_module.startswith("transfo-xl-")
    )
