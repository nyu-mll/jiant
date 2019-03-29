
class GAPScorer(object):
    """
    Container class for storing scores, and generating evaluation metrics.
    From Google. 
    Attributes:
    true_positives: Tally of true positives seen.
    false_positives: Tally of false positives seen.
    true_negatives: Tally of true negatives seen.
    false_negatives: Tally of false negatives seen.
    """
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def recall(self):
        """Calculates recall based on the observed scores.
        Returns:
          float, the recall.
        """
        numerator = self.true_positives
        denominator = self.true_positives + self.false_negatives
        return 100.0 * numerator / denominator if denominator else 0.0

    def precision(self):
        """Calculates precision based on the observed scores.
        Returns:
          float, the precision.
        """
        numerator = self.true_positives
        denominator = self.true_positives + self.false_positives
        return 100.0 * numerator / denominator if denominator else 0.0

    def f1(self):
        """Calculates F1 based on the observed scores.
        Returns:
          float, the F1 score.
        """
        recall = self.recall()
        precision = self.precision()

        numerator = 2 * precision * recall
        denominator = precision + recall
        return numerator / denominator if denominator else 0.0

    def get_metric(self, reset=False):
        recall = self.recall()
        precision = self.precision()
        f1 = self.f1()
        if reset:
            self.true_positives = 0
            self.false_positives = 0
            self.true_negatives = 0
            self.false_negatives = 0
        return f1

    def __call__(self, predictions, labels):
        """Score the system annotations against gold.
        Args:
        predictiosn: torch tensor of batch x 3
        system_annotations: batch x 3 Torch numpy list
        Returns:
        None
        """
        pred_indices = torch.max(predictions, dim=1)[1].view(-1, 1)
        num_classes = 3
        one_hot_logits = (pred_indices == torch.arange(num_classes).reshape(1, num_classes).cuda()).float()
        predictions = one_hot_logits[:,:2].cpu().numpy() 
        # get only the predictions for first and second label.
        labels = labels[:,:2].cpu().numpy()
        b_size = predictions.shape[0]
        for i in range(b_size):
            for j in range(2):
                pred = predictions[i][j].item()
                gold = labels[i][j].item()
                if gold and pred:
                  self.true_positives += 1
                elif not gold and pred:
                  self.false_positives += 1
                elif not gold and not pred:
                  self.true_negatives += 1
                elif gold and not pred:
                  self.false_negatives += 1
            return
