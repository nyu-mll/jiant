import copy
import glob
import itertools
import os
import shutil
from pathlib import Path

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.display as display
import jiant.utils.python.datastructures as datastructures
import jiant.utils.python.io as py_io
import jiant.utils.python.filesystem as filesystem
import jiant.utils.python.strings as strings


def download_xnli_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    xnli_temp_path = py_io.create_dir(task_data_base_path, "xnli_temp")
    download_utils.download_and_unzip(
        "https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip", xnli_temp_path,
    )
    full_val_data = py_io.read_jsonl(os.path.join(xnli_temp_path, "XNLI-1.0", "xnli.dev.jsonl"))
    val_data = datastructures.group_by(full_val_data, key_func=lambda elem: elem["language"])
    full_test_data = py_io.read_jsonl(os.path.join(xnli_temp_path, "XNLI-1.0", "xnli.test.jsonl"))
    test_data = datastructures.group_by(full_test_data, lambda elem: elem["language"])
    languages = sorted(list(val_data))
    for lang in languages:
        task_name = f"xnli_{lang}"
        task_data_path = py_io.create_dir(task_data_base_path, task_name)
        val_path = os.path.join(task_data_path, "val.jsonl")
        test_path = os.path.join(task_data_path, "test.jsonl")
        py_io.write_jsonl(data=val_data[lang], path=val_path)
        py_io.write_jsonl(data=test_data[lang], path=test_path)
        py_io.write_json(
            data={
                "task": "xnli",
                "paths": {"val": val_path, "test": test_path},
                "name": task_name,
                "kwargs": {"language": lang},
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(xnli_temp_path)


def download_pawsx_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    pawsx_temp_path = py_io.create_dir(task_data_base_path, "pawsx_temp")
    download_utils.download_and_untar(
        "https://storage.googleapis.com/paws/pawsx/x-final.tar.gz", pawsx_temp_path,
    )
    languages = sorted(os.listdir(os.path.join(pawsx_temp_path, "x-final")))
    for lang in languages:
        task_name = f"pawsx_{lang}"
        os.rename(
            src=os.path.join(pawsx_temp_path, "x-final", lang),
            dst=os.path.join(task_data_base_path, task_name),
        )
        paths_dict = {
            "val": os.path.join(task_data_base_path, task_name, "dev_2k.tsv"),
            "test": os.path.join(task_data_base_path, task_name, "test_2k.tsv"),
        }
        if lang == "en":
            paths_dict["train"] = os.path.join(task_data_base_path, task_name, "train.tsv")
            datastructures.set_dict_keys(paths_dict, ["train", "val", "test"])
        py_io.write_json(
            data={
                "task": "pawsx",
                "paths": paths_dict,
                "name": task_name,
                "kwargs": {"language": lang},
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(pawsx_temp_path)


def download_udpos_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    # UDPOS requires networkx==1.11

    def _read_one_file(file):
        # Adapted from https://github.com/JunjieHu/xtreme/blob/
        #              9fe0b142d0ee3eb7dd047ab86f12a76702e79bb4/utils_preprocess.py
        data = []
        sent, tag, lines = [], [], []
        for line in open(file, "r"):
            items = line.strip().split("\t")
            if len(items) != 10:
                num_empty = sum([int(w == "_") for w in sent])
                if num_empty == 0 or num_empty < len(sent) - 1:
                    data.append((sent, tag, lines))
                sent, tag, lines = [], [], []
            else:
                sent.append(items[1].strip())
                tag.append(items[3].strip())
                lines.append(line.strip())
                assert len(sent) == int(items[0]), "line={}, sent={}, tag={}".format(
                    line, sent, tag
                )
        return data

    def _remove_empty_space(data):
        # Adapted from https://github.com/google-research/xtreme/blob/
        #              522434d1aece34131d997a97ce7e9242a51a688a/utils_preprocess.py#L212
        new_data = {}
        for split in data:
            new_data[split] = []
            for sent, tag, lines in data[split]:
                new_sent = ["".join(w.replace("\u200c", "").split(" ")) for w in sent]
                lines = [line.replace("\u200c", "") for line in lines]
                assert len(" ".join(new_sent).split(" ")) == len(tag)
                new_data[split].append((new_sent, tag, lines))
        return new_data

    def check_file(file):
        # Adapted from https://github.com/google-research/xtreme/blob/
        #              522434d1aece34131d997a97ce7e9242a51a688a/utils_preprocess.py#L223
        for i, l in enumerate(open(file)):
            items = l.strip().split("\t")
            assert len(items[0].split(" ")) == len(items[1].split(" ")), "idx={}, line={}".format(
                i, l
            )

    def _write_files(data, output_dir, lang_, suffix):
        # Adapted from https://github.com/google-research/xtreme/blob/
        #              522434d1aece34131d997a97ce7e9242a51a688a/utils_preprocess.py#L228
        for split in data:
            if len(data[split]) > 0:
                prefix = os.path.join(output_dir, f"{split}-{lang_}")
                if suffix == "mt":
                    with open(prefix + ".mt.tsv", "w") as fout:
                        for idx, (sent, tag, _) in enumerate(data[split]):
                            newline = "\n" if idx != len(data[split]) - 1 else ""
                            fout.write("{}\t{}{}".format(" ".join(sent), " ".join(tag), newline))
                    check_file(prefix + ".mt.tsv")
                    print("    - finish checking " + prefix + ".mt.tsv")
                elif suffix == "tsv":
                    with open(prefix + ".tsv", "w") as fout:
                        for sidx, (sent, tag, _) in enumerate(data[split]):
                            for widx, (w, t) in enumerate(zip(sent, tag)):
                                newline = (
                                    ""
                                    if (sidx == len(data[split]) - 1) and (widx == len(sent) - 1)
                                    else "\n"
                                )
                                fout.write("{}\t{}{}".format(w, t, newline))
                            fout.write("\n")
                elif suffix == "conll":
                    with open(prefix + ".conll", "w") as fout:
                        for _, _, lines in data[split]:
                            for line in lines:
                                fout.write(line.strip() + "\n")
                            fout.write("\n")
                print(f"finish writing file to {prefix}.{suffix}")

    languages = (
        "af ar bg de el en es et eu fa fi fr he hi hu id it ja "
        "kk ko mr nl pt ru ta te th tl tr ur vi yo zh"
    ).split()
    udpos_temp_path = py_io.create_dir(task_data_base_path, "udpos_temp")
    download_utils.download_and_untar(
        "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/"
        "ud-treebanks-v2.5.tgz",
        udpos_temp_path,
    )
    download_utils.download_file(
        "https://raw.githubusercontent.com/google-research/xtreme/"
        "4fd68dc6b53796413e05dc3f3f73c2106f88ec57/third_party/ud-conversion-tools/lib/conll.py",
        os.path.join(udpos_temp_path, "conll.py"),
    )
    conll = filesystem.import_from_path(os.path.join(udpos_temp_path, "conll.py"))
    conllu_path_ls = sorted(glob.glob(os.path.join(udpos_temp_path, "*", "*", "*.conllu")))
    conll_path = os.path.join(udpos_temp_path, "conll")

    # === Convert conllu files to conll === #
    for input_path in display.tqdm(conllu_path_ls, desc="Convert conllu files to conll format"):
        input_path_fol, input_path_file = os.path.split(input_path)
        lang = input_path_file.split("_")[0]
        os.makedirs(os.path.join(conll_path, lang), exist_ok=True)
        output_path = os.path.join(
            conll_path, lang, strings.replace_suffix(input_path_file, "conllu", "conll")
        )
        pos_rank_precedence_dict = {
            "default": (
                "VERB NOUN PROPN PRON ADJ NUM ADV INTJ AUX ADP DET PART CCONJ SCONJ X PUNCT"
            ).split(" "),
            "es": "VERB AUX PRON ADP DET".split(" "),
            "fr": "VERB AUX PRON NOUN ADJ ADV ADP DET PART SCONJ CONJ".split(" "),
            "it": "VERB AUX ADV PRON ADP DET INTJ".split(" "),
        }

        if lang in pos_rank_precedence_dict:
            current_pos_precedence_list = pos_rank_precedence_dict[lang]
        else:
            current_pos_precedence_list = pos_rank_precedence_dict["default"]

        cio = conll.CoNLLReader()
        orig_treebank = cio.read_conll_u(input_path)
        modif_treebank = copy.copy(orig_treebank)

        for s in modif_treebank:
            s.filter_sentence_content(
                replace_subtokens_with_fused_forms=True,
                posPreferenceDict=current_pos_precedence_list,
                node_properties_to_remove=False,
                remove_deprel_suffixes=False,
                remove_arabic_diacritics=False,
            )

        cio.write_conll(
            list_of_graphs=modif_treebank,
            conll_path=Path(output_path),
            conllformat="conll2006",
            print_fused_forms=True,
            print_comments=False,
        )

    # === Convert conll to final format === #
    for lang in display.tqdm(languages, desc="Convert conll to final format"):
        task_name = f"udpos_{lang}"
        task_data_path = os.path.join(task_data_base_path, task_name)
        os.makedirs(task_data_path, exist_ok=True)
        all_examples = {k: [] for k in ["train", "val", "test"]}
        for path in sorted(glob.glob(os.path.join(conll_path, lang, "*.conll"))):
            examples = _read_one_file(path)
            if "train" in path:
                all_examples["train"] += examples
            elif "dev" in path:
                all_examples["val"] += examples
            elif "test" in path:
                all_examples["test"] += examples
            else:
                raise KeyError()
        all_examples = _remove_empty_space(all_examples)
        _write_files(
            data=all_examples, output_dir=task_data_path, lang_=lang, suffix="tsv",
        )
        paths_dict = {
            phase: os.path.join(task_data_path, f"{phase}-{lang}.tsv")
            for phase, phase_data in all_examples.items()
            if len(phase_data) > 0
        }
        py_io.write_json(
            data={
                "task": "udpos",
                "paths": paths_dict,
                "name": task_name,
                "kwargs": {"language": lang},
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(udpos_temp_path)


def download_panx_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    def _process_one_file(infile, outfile):
        lines = open(infile, "r").readlines()
        if lines[-1].strip() == "":
            lines = lines[:-1]
        with open(outfile, "w") as fout:
            for line in lines:
                items = line.strip().split("\t")
                if len(items) == 2:
                    label = items[1].strip()
                    idx = items[0].find(":")
                    if idx != -1:
                        token = items[0][idx + 1 :].strip()
                        fout.write(f"{token}\t{label}\n")
                else:
                    fout.write("\n")

    panx_temp_path = os.path.join(task_data_base_path, "panx_temp")
    zip_path = os.path.join(panx_temp_path, "AmazonPhotos.zip")
    assert os.path.exists(zip_path), (
        "Download AmazonPhotos.zip from"
        " https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN"
        f" and save it to {zip_path}"
    )
    download_utils.unzip_file(zip_path=zip_path, extract_location=panx_temp_path)
    languages = (
        "af ar bg bn de el en es et eu fa fi fr he hi hu id it ja jv ka "
        "kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh"
    ).split()
    for lang in languages:
        task_name = f"panx_{lang}"
        untar_path = os.path.join(panx_temp_path, "panx_dataset", lang)
        os.makedirs(untar_path, exist_ok=True)
        download_utils.untar_file(
            tar_path=os.path.join(panx_temp_path, "panx_dataset", f"{lang}.tar.gz"),
            extract_location=untar_path,
            delete=True,
        )
        task_data_path = os.path.join(task_data_base_path, task_name)
        os.makedirs(task_data_path, exist_ok=True)
        filename_dict = {"train": "train", "val": "dev", "test": "test"}
        paths_dict = {}
        for phase, filename in filename_dict.items():
            in_path = os.path.join(untar_path, filename)
            out_path = os.path.join(task_data_path, f"{phase}.tsv")
            if not os.path.exists(in_path):
                continue
            _process_one_file(infile=in_path, outfile=out_path)
            paths_dict[phase] = out_path
        py_io.write_json(
            data={
                "task": "panx",
                "paths": paths_dict,
                "name": task_name,
                "kwargs": {"language": lang},
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(os.path.join(panx_temp_path, "panx_dataset"))


def download_xquad_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    languages = "ar de el en es hi ru th tr vi zh".split()
    for lang in languages:
        task_name = f"xquad_{lang}"
        task_data_path = py_io.create_dir(task_data_base_path, task_name)
        path = os.path.join(task_data_path, "xquad.json")
        download_utils.download_file(
            url=f"https://raw.githubusercontent.com/deepmind/xquad/master/xquad.{lang}.json",
            file_path=path,
        )
        py_io.write_json(
            data={
                "task": "xquad",
                "paths": {"val": path},
                "name": task_name,
                "kwargs": {"language": lang},
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )


def download_mlqa_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    mlqa_temp_path = py_io.create_dir(task_data_base_path, "mlqa_temp")
    download_utils.download_and_unzip(
        "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip", mlqa_temp_path,
    )
    languages = "ar de en es hi vi zh".split()
    for lang1, lang2 in itertools.product(languages, languages):
        task_name = f"mlqa_{lang1}_{lang2}"
        task_data_path = py_io.create_dir(task_data_base_path, task_name)
        val_path = os.path.join(task_data_path, f"dev-context-{lang1}-question-{lang2}.json")
        os.rename(
            src=os.path.join(
                mlqa_temp_path, "MLQA_V1", "dev", f"dev-context-{lang1}-question-{lang2}.json"
            ),
            dst=val_path,
        )
        test_path = os.path.join(task_data_path, f"test-context-{lang1}-question-{lang2}.json")
        os.rename(
            src=os.path.join(
                mlqa_temp_path, "MLQA_V1", "test", f"test-context-{lang1}-question-{lang2}.json"
            ),
            dst=test_path,
        )
        py_io.write_json(
            data={
                "task": "mlqa",
                "paths": {"val": val_path, "test": test_path},
                "kwargs": {"context_language": lang1, "question_language": lang2},
                "name": task_name,
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(mlqa_temp_path)


def download_tydiqa_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    tydiqa_temp_path = py_io.create_dir(task_data_base_path, "tydiqa_temp")
    full_train_path = os.path.join(tydiqa_temp_path, "tydiqa-goldp-v1.1-train.json")
    download_utils.download_file(
        "https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json", full_train_path,
    )
    download_utils.download_and_untar(
        "https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.tgz", tydiqa_temp_path,
    )
    languages_dict = {
        "arabic": "ar",
        "bengali": "bn",
        "english": "en",
        "finnish": "fi",
        "indonesian": "id",
        "korean": "ko",
        "russian": "ru",
        "swahili": "sw",
        "telugu": "te",
    }

    # Split train data
    data = py_io.read_json(full_train_path)
    lang2data = {lang: [] for lang in languages_dict.values()}
    for doc in data["data"]:
        for par in doc["paragraphs"]:
            context = par["context"]
            for qa in par["qas"]:
                question = qa["question"]
                question_id = qa["id"]
                example_lang = languages_dict[question_id.split("-")[0]]
                q_id = question_id.split("-")[-1]
                for answer in qa["answers"]:
                    a_start, a_text = answer["answer_start"], answer["text"]
                    a_end = a_start + len(a_text)
                    assert context[a_start:a_end] == a_text
                lang2data[example_lang].append(
                    {
                        "paragraphs": [
                            {
                                "context": context,
                                "qas": [
                                    {"answers": qa["answers"], "question": question, "id": q_id}
                                ],
                            }
                        ]
                    }
                )

    for full_lang, lang in languages_dict.items():
        task_name = f"tydiqa_{lang}"
        task_data_path = py_io.create_dir(task_data_base_path, task_name)
        train_path = os.path.join(task_data_path, f"tydiqa.{lang}.train.json")
        py_io.write_json(
            data=data, path=train_path,
        )
        val_path = os.path.join(task_data_path, f"tydiqa.{lang}.dev.json")
        os.rename(
            src=os.path.join(
                tydiqa_temp_path, "tydiqa-goldp-v1.1-dev", f"tydiqa-goldp-dev-{full_lang}.json"
            ),
            dst=val_path,
        )
        py_io.write_json(
            data={
                "task": "tydiqa",
                "paths": {"train": train_path, "val": val_path},
                "kwargs": {"language": lang},
                "name": task_name,
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(tydiqa_temp_path)


def download_bucc2018_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    bucc2018_temp_path = py_io.create_dir(task_data_base_path, "bucc_temp")
    languages = "de fr ru zh".split()
    for lang in languages:
        download_utils.download_and_untar(
            f"https://comparable.limsi.fr/bucc2018/bucc2018-{lang}-en.training-gold.tar.bz2",
            bucc2018_temp_path,
        )
        download_utils.download_and_untar(
            f"https://comparable.limsi.fr/bucc2018/bucc2018-{lang}-en.sample-gold.tar.bz2",
            bucc2018_temp_path,
        )
    for lang in languages:
        task_name = f"bucc2018_{lang}"
        task_data_path = py_io.create_dir(task_data_base_path, task_name)
        val_eng_path = os.path.join(task_data_path, f"{lang}-en.dev.en")
        val_other_path = os.path.join(task_data_path, f"{lang}-en.dev.{lang}")
        val_labels_path = os.path.join(task_data_path, f"{lang}-en.dev.gold")
        test_eng_path = os.path.join(task_data_path, f"{lang}-en.test.en")
        test_other_path = os.path.join(task_data_path, f"{lang}-en.test.{lang}")
        # sample -> dev
        # training -> test (yup, it's weird)
        os.rename(
            src=os.path.join(bucc2018_temp_path, "bucc2018", f"{lang}-en", f"{lang}-en.sample.en"),
            dst=val_eng_path,
        )
        os.rename(
            src=os.path.join(
                bucc2018_temp_path, "bucc2018", f"{lang}-en", f"{lang}-en.sample.{lang}"
            ),
            dst=val_other_path,
        )
        os.rename(
            src=os.path.join(
                bucc2018_temp_path, "bucc2018", f"{lang}-en", f"{lang}-en.sample.gold"
            ),
            dst=val_labels_path,
        )
        os.rename(
            src=os.path.join(
                bucc2018_temp_path, "bucc2018", f"{lang}-en", f"{lang}-en.training.en"
            ),
            dst=test_eng_path,
        )
        os.rename(
            src=os.path.join(
                bucc2018_temp_path, "bucc2018", f"{lang}-en", f"{lang}-en.training.{lang}"
            ),
            dst=test_other_path,
        )
        py_io.write_json(
            data={
                "task": "bucc2018",
                "paths": {
                    "val": {
                        "eng": val_eng_path,
                        "other": val_other_path,
                        "labels": val_labels_path,
                    },
                    "test": {"eng": test_eng_path, "other": test_other_path},
                },
                "kwargs": {"language": lang},
                "name": task_name,
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(bucc2018_temp_path)


def download_tatoeba_data_and_write_config(task_data_base_path: str, task_config_base_path: str):
    tatoeba_temp_path = py_io.create_dir(task_data_base_path, "tatoeba_temp")
    download_utils.download_and_unzip(
        "https://github.com/facebookresearch/LASER/archive/master.zip", tatoeba_temp_path,
    )
    languages_dict = {
        "afr": "af",
        "ara": "ar",
        "bul": "bg",
        "ben": "bn",
        "deu": "de",
        "ell": "el",
        "spa": "es",
        "est": "et",
        "eus": "eu",
        "pes": "fa",
        "fin": "fi",
        "fra": "fr",
        "heb": "he",
        "hin": "hi",
        "hun": "hu",
        "ind": "id",
        "ita": "it",
        "jpn": "ja",
        "jav": "jv",
        "kat": "ka",
        "kaz": "kk",
        "kor": "ko",
        "mal": "ml",
        "mar": "mr",
        "nld": "nl",
        "por": "pt",
        "rus": "ru",
        "swh": "sw",
        "tam": "ta",
        "tel": "te",
        "tha": "th",
        "tgl": "tl",
        "tur": "tr",
        "urd": "ur",
        "vie": "vi",
        "cmn": "zh",
        "eng": "en",
    }
    raw_base_path = os.path.join(tatoeba_temp_path, "LASER-master", "data", "tatoeba", "v1")
    for full_lang, lang in languages_dict.items():
        task_name = f"tatoeba_{lang}"
        if lang == "en":
            continue
        task_data_path = py_io.create_dir(task_data_base_path, task_name)
        eng_src = os.path.join(raw_base_path, f"tatoeba.{full_lang}-eng.eng")
        other_src = os.path.join(raw_base_path, f"tatoeba.{full_lang}-eng.{full_lang}")
        eng_out = os.path.join(task_data_path, f"{lang}-en.en")
        other_out = os.path.join(task_data_path, f"{lang}-en.{lang}")
        labels_out = os.path.join(task_data_path, f"{lang}-en.labels")
        tgts = [line.strip() for line in py_io.read_file_lines(eng_src)]
        os.rename(src=other_src, dst=other_out)
        idx = range(len(tgts))
        data = zip(tgts, idx)

        # Tatoeba is a retrieval dataset where you have a set of sentences in English and another
        # set in another language, and you need to match them. It also doesn't have training
        # data, so it's pretty much evaluation only. However, the dataset is distributed with the
        # sentences in order, i.e. the retrieval pairing is the sentence order.
        #
        # The XTREME authors intentionally scramble the order by sorting one of the two
        # sets alphabetically. We're following their recipe, but also retaining the labels for
        # internal scoring.
        with open(eng_out, "w") as ftgt, open(labels_out, "w") as flabels:
            for t, i in sorted(data, key=lambda x: x[0]):
                ftgt.write(f"{t}\n")
                flabels.write(f"{i}\n")
        py_io.write_json(
            data={
                "task": "tatoeba",
                "paths": {"eng": eng_out, "other": other_out, "labels_path": labels_out},
                "kwargs": {"language": lang},
                "name": task_name,
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(tatoeba_temp_path)


def download_xtreme_data_and_write_config(
    task_name: str, task_data_base_path: str, task_config_base_path: str
):
    if task_name == "xnli":
        download_xnli_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    elif task_name == "pawsx":
        download_pawsx_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    elif task_name == "udpos":
        download_udpos_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    elif task_name == "panx":
        download_panx_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    elif task_name == "xquad":
        download_xquad_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    elif task_name == "mlqa":
        download_mlqa_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    elif task_name == "tydiqa":
        download_tydiqa_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    elif task_name == "bucc2018":
        download_bucc2018_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    elif task_name == "tatoeba":
        download_tatoeba_data_and_write_config(
            task_data_base_path=task_data_base_path, task_config_base_path=task_config_base_path,
        )
    else:
        raise KeyError(task_name)
