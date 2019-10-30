"""Task definitions for acceptablity probing tasks."""
import logging as log
import os

from jiant.utils.data_loaders import load_tsv
from jiant.tasks.registry import register_task
from jiant.tasks.tasks import SingleClassificationTask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@register_task("acceptability-def", "FunctionWordsProbing/definiteness/")
@register_task("acceptability-conj", "FunctionWordsProbing/coordinating-conjunctions/")
@register_task("acceptability-wh", "FunctionWordsProbing/whwords/")
@register_task("acceptability-eos", "FunctionWordsProbing/eos/")
class AcceptabilityProbingTask(SingleClassificationTask):
    """ Task class for A-type Probing Task
    This probing task only have evaluation set, need to share classifier with NLI-type task.
    At present, 4 tasks are registered under this class,
    acceptability-def tests model's understanding of definiteness
    acceptability-conj tests that of coordinating conjunctions
    acceptability-wh tests that of wh-words
    acceptability-eos tests that of EOS
    """

    def __init__(self, path, max_seq_len, name, fold_no=1):
        super(AcceptabilityProbingTask, self).__init__(name, max_seq_len)
        self.load_data(path, max_seq_len, fold_no)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.val_metric = "%s_acc_f1" % self.name
        self.val_metric_decreases = False
        self.scorer1 = CategoricalAccuracy()
        self.scorer2 = F1Measure(1)

    def load_data(self, path, max_seq_len, fold_no):
        tr_data = load_tsv(
            os.path.join(path, "fold{}/train.tsv".format(fold_no)),
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            targ_idx=2,
            skip_rows=0,
        )
        val_data = load_tsv(
            os.path.join(path, "fold{}/dev.tsv".format(fold_no)),
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            targ_idx=2,
            skip_rows=0,
        )
        te_data = load_tsv(
            os.path.join(path, "fold{}/test.tsv".format(fold_no)),
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            targ_idx=2,
            skip_rows=0,
        )

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info(
            "\tFinished loading acceptability probing {} data (fold{}).".format(self.name, fold_no)
        )

    def get_metrics(self, reset=False):
        acc = self.scorer1.get_metric(reset)
        pcs, rcl, f1 = self.scorer2.get_metric(reset)
        return {"acc_f1": (acc + f1) / 2, "accuracy": acc, "f1": f1}
