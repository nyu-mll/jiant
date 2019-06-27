class GenderParity:
    """
  Gender parity metric from Gender parity metric from https://github.com/decompositional-semantics-initiative/DNC.
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
        predictiosn: list of dicts of type
          sent2_str: str, hypothesis sentence,
          sent1_str: str, context sentence ,
          pair_id: int, 

        Returns
        -------------------
        None
        """
        for idx in range(int(len(predictions) / 2)):
            pred1 = predictions[idx * 2]
            pred2 = predictions[(idx * 2) + 1]
                assert pred1["sent2_str"] == pred2["sent2_str"], (
                    "Mismatched hypotheses for ids %s and %s"
                    % (str(pred1["pair_id"]), str(pred2["pair_id"]))
                )
  
            if pred1["preds"] == pred2["preds"]:
                self.same_preds += 1
            else:
                self.diff_preds += 1
