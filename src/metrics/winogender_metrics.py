class GenderParity:
    """
  Gender parity metric from [Gender Bias in Coreference Resolution](https://arxiv.org/pdf/1804.09301.pdf).
  """

    def __init__(self):
        self.same_preds = 0.0
        self.diff_preds = 0.0

    def get_metric(self, reset=False):
        gender_parity = 100 * (float(self.same_preds) / float(self.same_preds + self.diff_preds))
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
            pred1 = predictions[idx * 2]
            pred2 = predictions[(idx * 2) + 1]
            try:
                assert pred1["sent2_str"] == pred2["sent2_str"], (
                    "Mismatched hypotheses for ids %s and %s"
                    % (str(pred1["pair_id"]), str(pred2["pair_id"]))
                )
            except:
                import pdb

                pdb.set_trace()
            if pred1["preds"] == pred2["preds"]:
                self.same_preds += 1
            else:
                self.diff_preds += 1
