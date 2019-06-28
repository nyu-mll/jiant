import torch
from torch import nn


class PairClassifier(nn.Module):
    """ Thin wrapper around a set of modules.
    For sentence pair classification.
    Pooler specifies how to aggregate inputted sequence of vectors.
    Also allows for use of specific token representations to be addded to the overall
    representation
    """

    def __init__(self, pooler, classifier, attn=None):
        super(PairClassifier, self).__init__()
        self.pooler = pooler
        self.classifier = classifier
        self.attn = attn

    def forward(self, s1, s2, mask1, mask2, idx1=[], idx2=[]):
        """ s1, s2: sequences of hidden states corresponding to sentence 1,2
            mask1, mask2: binary mask corresponding to non-pad elements
            idx{1,2}: indexes of particular tokens to extract in sentence {1, 2}
                and append to the representation, e.g. for WiC
        """
        mask1 = mask1.squeeze(-1) if len(mask1.size()) > 2 else mask1
        mask2 = mask2.squeeze(-1) if len(mask2.size()) > 2 else mask2
        if self.attn is not None:
            s1, s2 = self.attn(s1, s2, mask1, mask2)
        emb1 = self.pooler(s1, mask1)
        emb2 = self.pooler(s2, mask2)

        s1_ctx_embs = []
        for idx in [i.long() for i in idx1]:
            if len(idx.shape) == 1:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 2:
                idx = idx.unsqueeze(-1).expand([-1, -1, s1.size(-1)])
            s1_ctx_emb = s1.gather(dim=1, index=idx)
            s1_ctx_embs.append(s1_ctx_emb.squeeze(dim=1))
        emb1 = torch.cat([emb1] + s1_ctx_embs, dim=-1)

        s2_ctx_embs = []
        for idx in [i.long() for i in idx2]:
            if len(idx.shape) == 1:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 2:
                idx = idx.unsqueeze(-1).expand([-1, -1, s2.size(-1)])
            s2_ctx_emb = s2.gather(dim=1, index=idx)
            s2_ctx_embs.append(s2_ctx_emb.squeeze(dim=1))
        emb2 = torch.cat([emb2] + s2_ctx_embs, dim=-1)

        pair_emb = torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], 1)
        logits = self.classifier(pair_emb)
        return logits
