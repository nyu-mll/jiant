import hashlib
import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import (
    construct_single_input_tokens_and_segment_ids,
    create_input_set_from_tokens_and_segments,
)
from jiant.utils.python.io import read_file, read_file_lines


@dataclass
class Example(BaseExample):
    guid: str
    is_english: bool
    text: str
    text_hash: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            is_english=self.is_english,
            text_tokens=tokenizer.tokenize(self.text),
            text_hash=self.text_hash,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    is_english: bool
    text_tokens: List
    text_hash: str

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.text_tokens, tokenizer=tokenizer, feat_spec=feat_spec,
        )
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            is_english=self.is_english,
            tokens=unpadded_inputs.unpadded_tokens,
            text_hash=self.text_hash,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    is_english: bool
    tokens: list
    text_hash: str


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    is_english: torch.BoolTensor
    tokens: list
    text_hash: list
    guid: list


class Bucc2018Task(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.EMBEDDING

    def __init__(self, name, path_dict, language):
        super().__init__(name=name, path_dict=path_dict)
        self.language = language

    def get_train_examples(self):
        raise RuntimeError("This task does not support train examples")

    def get_val_examples(self):
        return self._get_examples(phase="val")

    def get_test_examples(self):
        return self._get_examples(phase="test")

    def get_val_labels(self):
        return read_file(self.path_dict["val"]["labels"]).strip().splitlines()

    def _get_examples(self, phase):
        eng_examples = self._create_examples(
            lines=read_file_lines(self.path_dict[phase]["eng"]), is_english=True, set_type=phase,
        )
        other_examples = self._create_examples(
            lines=read_file_lines(self.path_dict[phase]["other"]), is_english=False, set_type=phase,
        )
        return eng_examples + other_examples

    @classmethod
    def _create_examples(cls, lines, is_english, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            idx, text = line.split("\t")
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, idx),
                    is_english=is_english,
                    text=text.strip(),
                    text_hash=hashlib.sha1(text.strip().encode("utf-8")).hexdigest(),
                )
            )
        return examples


# noinspection PyUnboundLocalVariable
def mine_bitext(
    x,
    y,
    src_inds,
    trg_inds,
    mode="mine",
    retrieval="max",
    margin="ratio",
    threshold=0,
    neighborhood=4,
    use_gpu=False,
    dist="cosine",
    use_shift_embeds=False,
):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    import faiss

    src_orig_inds = np.arange(len(x))
    trg_orig_inds = np.arange(len(y))
    out_ls = []

    x = unique_embeddings(x, src_inds)
    y = unique_embeddings(y, trg_inds)
    if dist == "cosine":
        faiss.normalize_L2(x)
        faiss.normalize_L2(y)

    if use_shift_embeds:
        x2y, y2x = shift_embeddings(x, y)

    # calculate knn in both directions
    if retrieval != "bwd":
        if use_shift_embeds:
            # project x to y space, and search k-nn ys for each x
            x2y_sim, x2y_ind = knn(x2y, y, min(y.shape[0], neighborhood), use_gpu, dist)
            x2y_mean = x2y_sim.mean(axis=1)
        else:
            x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], neighborhood), use_gpu, dist)
            x2y_mean = x2y_sim.mean(axis=1)

    if retrieval != "fwd":
        if use_shift_embeds:
            y2x_sim, y2x_ind = knn(y2x, x, min(x.shape[0], neighborhood), use_gpu, dist)
            y2x_mean = y2x_sim.mean(axis=1)
        else:
            y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], neighborhood), use_gpu, dist)
            y2x_mean = y2x_sim.mean(axis=1)

    # margin function
    if margin == "absolute":
        # noinspection PyUnusedLocal
        def margin(a, b):
            return a

    elif margin == "distance":

        def margin(a, b):
            return a - b

    else:  # margin == 'ratio':

        def margin(a, b):
            return a / b

    if mode == "search":
        scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
        best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

        for i in src_inds:
            out_ls.append(trg_orig_inds[best[i]])

    elif mode == "score":
        for i, j in zip(src_inds, trg_inds):
            s = score(x[i], y[j], x2y_mean[i], y2x_mean[j], margin)
            out_ls.append((s, src_orig_inds[i], trg_orig_inds[j]))

    elif mode == "mine":
        if use_shift_embeds:
            fwd_scores = score_candidates(x2y, y, x2y_ind, x2y_mean, y2x_mean, margin)
            bwd_scores = score_candidates(y2x, x, y2x_ind, y2x_mean, x2y_mean, margin)
        else:
            fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
            bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin)
        fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
        bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]
        if retrieval == "fwd":
            for i, j in enumerate(fwd_best):
                out_ls.append((fwd_scores[i].max(), src_orig_inds[i], trg_orig_inds[j]))
        if retrieval == "bwd":
            for j, i in enumerate(bwd_best):
                out_ls.append((bwd_scores[j].max(), src_orig_inds[i], trg_orig_inds[j]))
        if retrieval == "intersect":
            for i, j in enumerate(fwd_best):
                if bwd_best[j] == i:
                    out_ls.append((fwd_scores[i].max(), src_orig_inds[i], trg_orig_inds[j]))
        if retrieval == "max":
            indices = np.stack(
                (
                    np.concatenate((np.arange(x.shape[0]), bwd_best)),
                    np.concatenate((fwd_best, np.arange(y.shape[0]))),
                ),
                axis=1,
            )
            # noinspection PyArgumentList
            scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
            seen_src, seen_trg = set(), set()
            for i in np.argsort(-scores):
                src_ind, trg_ind = indices[i]
                if src_ind not in seen_src and trg_ind not in seen_trg:
                    seen_src.add(src_ind)
                    seen_trg.add(trg_ind)
                    if scores[i] > threshold:
                        out_ls.append((scores[i], src_orig_inds[src_ind], trg_orig_inds[trg_ind]))
    return out_ls


def bucc_eval(candidates2score, gold, threshold=None):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    if threshold is not None:
        print(" - using threshold {}".format(threshold))
    else:
        threshold = bucc_optimize(candidates2score, gold)

    gold = set(gold)
    bitexts = bucc_extract(candidates2score, threshold)
    ncorrect = len(gold.intersection(bitexts))
    if ncorrect > 0:
        precision = ncorrect / len(bitexts)
        recall = ncorrect / len(gold)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = recall = f1 = 0
    return {
        "best-threshold": threshold,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def bucc_optimize(candidate2score, gold):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    items = sorted(candidate2score.items(), key=lambda x: -x[1])
    ngold = len(gold)
    nextract = ncorrect = 0
    threshold = 0
    best_f1 = 0
    for i in range(len(items)):
        nextract += 1
        if items[i][0] in gold:
            ncorrect += 1
        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / ngold
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                threshold = (items[i][1] + items[i + 1][1]) / 2
    return threshold


def bucc_extract(cand2score, th):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    bitexts = []
    for (src, trg), score_ in cand2score.items():
        if score_ >= th:
            bitexts.append((src, trg))
    return bitexts


def unique_embeddings(emb, ind):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    aux = {j: i for i, j in enumerate(ind)}
    return emb[[aux[i] for i in range(len(aux))]]


def shift_embeddings(x, y):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    delta = x.mean(axis=0) - y.mean(axis=0)
    x2y = x - delta
    y2x = y + delta
    return x2y, y2x


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin, dist="cosine"):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin, dist)
    return scores


def score(x, y, fwd_mean, bwd_mean, margin, dist="cosine"):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    if dist == "cosine":
        return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)
    else:
        l2 = ((x - y) ** 2).sum()
        sim = 1 / (1 + l2)
        return margin(sim, (fwd_mean + bwd_mean) / 2)


def knn(x, y, k, use_gpu, dist="cosine"):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    return knn_gpu(x, y, k) if use_gpu else knn_cpu(x, y, k, dist)


def knn_gpu(x, y, k, mem=5 * 1024 * 1024 * 1024):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    import faiss

    dim = x.shape[1]
    batch_size = mem // (dim * 4)
    sim = np.zeros((x.shape[0], k), dtype=np.float32)
    ind = np.zeros((x.shape[0], k), dtype=np.int64)
    for xfrom in range(0, x.shape[0], batch_size):
        xto = min(xfrom + batch_size, x.shape[0])
        bsims, binds = [], []
        for yfrom in range(0, y.shape[0], batch_size):
            yto = min(yfrom + batch_size, y.shape[0])
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(y[yfrom:yto])
            bsim, bind = idx.search(x[xfrom:xto], min(k, yto - yfrom))
            bsims.append(bsim)
            binds.append(bind + yfrom)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(xfrom, xto):
            for j in range(k):
                sim[i, j] = bsims[i - xfrom, aux[i - xfrom, j]]
                ind[i, j] = binds[i - xfrom, aux[i - xfrom, j]]
    return sim, ind


def knn_cpu(x, y, k, dist="cosine"):
    # Adapted From: https://github.com/google-research/xtreme/blob/
    #               522434d1aece34131d997a97ce7e9242a51a688a/third_party/utils_retrieve.py
    import faiss

    # x: query, y: database
    dim = x.shape[1]
    if dist == "cosine":
        idx = faiss.IndexFlatIP(dim)
    else:
        idx = faiss.IndexFlatL2(dim)
    idx.add(y)
    sim, ind = idx.search(x, k)

    if dist != "cosine":
        sim = 1 / (1 + sim)
    return sim, ind


def get_unique_lines(text_hashes):
    """Get the unique lines out of a list of text-hashes

    Args:
        text_hashes (list): A list of (hashes of) strings

    Returns:
        unique_indices (List): List (of the same length as text_hashes) indicating, for each
                               element of text_hash, the index of the corresponding entry in
                               unique_text_hashes
                               (i.e. "Which unique text-hash does this correspond to?")
        unique_text_hashes (List): List of unique elements of text_hashes
    """
    unique_text_hashes = []
    unique_indices = []
    unique_lookup = {}
    for text_hash in text_hashes:
        if text_hash not in unique_lookup:
            unique_lookup[text_hash] = len(unique_lookup)
            unique_text_hashes.append(text_hash)
        unique_indices.append(unique_lookup[text_hash])
    return unique_indices, unique_text_hashes


def create_ids_map(inds, guids):
    ids_map = {}
    for guid, idx in zip(guids, inds):
        if idx not in ids_map:
            ids_map[idx] = []
        ids_map[idx].append(guid)
    return ids_map
