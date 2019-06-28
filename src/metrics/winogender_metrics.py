class GenderParity:
    """
  Gender parity metric from https://github.com/decompositional-semantics-initiative/DNC.
  """

    def __init__(self):
        self.same_preds = 0.0
        self.diff_preds = 0.0

    def get_metric(self, reset=False):
        gender_parity = float(self.same_preds) / float(self.same_preds + self.diff_preds)
        if reset:
            self.same_preds = 0.0
            self.diff_preds = 0.0
        return gender_parity

    def __call__(self, predictions):
        """
        Calculate gender parity. 
        Parameters
        -------------------
        predictions: list of dicts with fields
            sent2_str: str, hypothesis sentence,
            sent1_str: str, context sentence,
            pair_id: int

        Returns
        -------------------
        None
        """
        for idx in range(int(len(predictions) / 2)):
            pred1 = predictions[idx * 2]
            pred2 = predictions[(idx * 2) + 1]
            if pred1["sent2_str"] != pred2["sent2_str"]:
                import pdb

                pdb.set_trace()

            if pred1["preds"] == pred2["preds"]:
                self.same_preds += 1
            else:
                self.diff_preds += 1
