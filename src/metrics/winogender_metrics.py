
class GenderParity:
  """
  Gender parity metric from [Gender Bias in Coreference Resolution](https://arxiv.org/pdf/1804.09301.pdf).
  """

  def __init__(self):
      self.same_preds = 0.0
      self.diff_preds = 0.0

  def get_metric(self, reset=False):
      gender_parity = 100 * self.same_pred / (self.same_pred + self.diff_pred)
      if reset:
          self.same_preds = 0.0
          self.diff_preds = 0.0
      return gender_parity

  def __call__(self, predictions):
      """
      Calculate gender parity. 
      Parameters
      -------------------
      predictiosn: list of dicts 

      Returns
      -------------------
      None
      """
      assert len(predictions) == 464
      for idx in range(int(len(predictions) / 2)):
          pred1 = predictions[idx]
          pred2 = predictions[idx + 1]
          try:
              assert pred1["sent1_str"] == pred2["sent1_str"], (
                  "Mismatched hypotheses for ids %s and %s"
                  % (str(pred1["pair_id"]), str(pred2["pair_id"]))
              )
          except:
              import pdb; pdb.set_trace()
          if pred1["preds"] == pred2["preds"]:
              self.same_preds += 1
          else:
              self.diff_preds += 1
