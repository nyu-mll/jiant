import itertools
import os
import shutil

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.python.datastructures as datastructures
import jiant.utils.python.io as py_io


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
            paths_dict["train"] = (os.path.join(task_data_base_path, task_name, "train.tsv"),)
            datastructures.set_dict_keys(paths_dict, ["train", "val", "test"])
        py_io.write_json(
            data={"task": "pawsx", "paths": paths_dict, "name": task_name},
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    shutil.rmtree(pawsx_temp_path)


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
    version = data["version"]
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
            data={"data": data, "version": version}, path=train_path,
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
                "task": "tatoeba",
                "paths": {
                    "val": {
                        "eng": val_eng_path,
                        "other": val_other_path,
                        "labels": val_labels_path,
                    },
                    "test": {"eng": test_eng_path, "other": test_other_path},
                },
                "name": task_name,
            },
            path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )


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
    elif task_name in ["udpos", "panx"]:
        raise NotImplementedError(task_name)
    else:
        raise KeyError(task_name)
