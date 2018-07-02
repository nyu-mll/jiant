import ipdb as pdb
import numpy as np
from overrides import overrides
from allennlp.training.metrics.metric import Metric
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

@Metric.register("correlation")
class Correlation(Metric):
    """Aggregate predictions, then calculate specified correlation"""
    def __init__(self, corr_type):
        self._predictions = []
        self._labels = []
        if corr_type == 'pearson':
            corr_fn = pearsonr
        elif corr_type == 'spearman':
            corr_fn = spearmanr
        elif corr_type == 'matthews':
            corr_fn = matthews_corrcoef
        else:
            raise ValueError("Correlation type not supported")
        self._corr_fn = corr_fn
        self.corr_type = corr_type

    def __call__(self, labels, predictions):
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        self._predictions += predictions
        self._labels += labels

    def get_metric(self, reset=False):
        correlation = self._corr_fn(self._labels, self._predictions)
        if self.corr_type in ['pearson', 'spearman']:
            correlation = correlation[0]
        if reset:
            self.reset()
        return correlation

    @overrides
    def reset(self):
        self._predictions = []
        self._labels = []
