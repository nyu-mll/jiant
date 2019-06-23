import torch

class GenderParity:

  def __init__(self):
    self.same_preds = 0.
    self.diff_preds = 0.

  def get_metric(self, reset=False):
        gender_parity = (100 * same_pred / (same_pred + diff_pred))
        if reset:
            self.same_preds = 0.
            self.diff_preds = 0.
        return gender_parity

  def __call__(self, predictions, labels):
        """Score the system annotations against gold.
        Parameteres
        -------------------
        predictiosn: torch tensor of batch x 2
        label: batch x 2 Torch numpy list

        Returns:
        -------------------
        None
        """
        pred_indices = torch.max(predictions, dim=1)[1].view(-1, 1)
        num_classes = 2
        one_hot_logits = (pred_indices == torch.arange(num_classes).reshape(1, num_classes).cuda()).float()
        predictions = one_hot_logits[:,:2].cpu().numpy()
        labels = labels[:,:2].cpu().numpy()
        b_size = predictions.shape[0]
        for idx in range(len(predictions)/2):
          pred1 = predictions[idx]
          pred2 = predictions[idx+1]
          assert pred1['hypothesis'] == pred2['hypothesis'], "Mismatched hypotheses for ids  %s and %s" % (str(pred1['pair_id']), str(pred2['pair_id']))
          if pred1 == pred2:
            self.same_preds += 1
          else:
            self.diff_preds += 1
