import abc

import torch
import torch.nn as nn

import transformers
from jiant.ext.allennlp import SelfAttentiveSpanExtractor

"""
In HuggingFace/others, these heads differ slightly across different encoder models.
We're going to abstract away from that and just choose one implementation.
"""


class BaseHead(nn.Module, metaclass=abc.ABCMeta):
    pass


class ClassificationHead(BaseHead):
    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        """From RobertaClassificationHead"""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pooled):
        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class RegressionHead(BaseHead):
    def __init__(self, hidden_size, hidden_dropout_prob):
        """From RobertaClassificationHead"""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, pooled):
        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        scores = self.out_proj(x)
        return scores


class SpanComparisonHead(BaseHead):
    def __init__(self, hidden_size, hidden_dropout_prob, num_spans, num_labels):
        """From RobertaForSpanComparisonClassification"""
        super().__init__()
        self.num_spans = num_spans
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.span_attention_extractor = SelfAttentiveSpanExtractor(hidden_size)
        self.classifier = nn.Linear(hidden_size * self.num_spans, self.num_labels)

    def forward(self, unpooled, spans):
        span_embeddings = self.span_attention_extractor(unpooled, spans)
        span_embeddings = span_embeddings.view(-1, self.num_spans * self.hidden_size)
        span_embeddings = self.dropout(span_embeddings)
        logits = self.classifier(span_embeddings)
        return logits


class TokenClassificationHead(BaseHead):
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob):
        """From RobertaForTokenClassification"""
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, unpooled):
        unpooled = self.dropout(unpooled)
        logits = self.classifier(unpooled)
        return logits


class QAHead(BaseHead):
    def __init__(self, hidden_size):
        """From RobertaForQuestionAnswering"""
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, unpooled):
        logits = self.qa_outputs(unpooled)
        # bs x seq_len x 2
        logits = logits.permute(0, 2, 1)
        # bs x 2 x seq_len x 1
        return logits


class BaseMLMHead(BaseHead, metaclass=abc.ABCMeta):
    pass


class BertMLMHead(BaseMLMHead):
    """From BertOnlyMLMHead, BertLMPredictionHead, BertPredictionHeadTransform"""

    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12, hidden_act="gelu"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transformers.modeling_bert.ACT2FN[hidden_act]
        self.LayerNorm = transformers.modeling_bert.BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)

        # Need a link between the two variables so that the bias is correctly resized with
        # `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, unpooled):
        hidden_states = self.dense(unpooled)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        logits = self.decoder(hidden_states) + self.bias
        return logits


class RobertaMLMHead(BaseMLMHead):
    """From RobertaLMHead"""

    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = transformers.modeling_bert.BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)

        # Need a link between the two variables so that the bias is correctly resized with
        # `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, unpooled):
        x = self.dense(unpooled)
        x = transformers.modeling_bert.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        logits = self.decoder(x) + self.bias
        return logits


class AlbertMLMHead(nn.Module):
    """From AlbertMLMHead"""

    def __init__(self, hidden_size, embedding_size, vocab_size, hidden_act="gelu"):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(embedding_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, vocab_size)
        self.activation = transformers.modeling_bert.ACT2FN[hidden_act]

        # Need a link between the two variables so that the bias is correctly resized with
        # `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, unpooled):
        hidden_states = self.dense(unpooled)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        logits = hidden_states + self.bias
        return logits


class AbstractPoolerHead(nn.Module):
    pass


class MeanPoolerHead(AbstractPoolerHead):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, unpooled, input_mask):
        # [batch_size, length, hidden_dim]
        assert len(unpooled.shape) == 3
        # [batch_size, length]
        assert len(input_mask.shape) == 2
        lengths = input_mask.sum(dim=1).float()
        summed = (unpooled * input_mask.float().unsqueeze(2)).sum(1)
        return summed / lengths.unsqueeze(1)


class FirstPoolerHead(AbstractPoolerHead):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, unpooled):
        # [batch_size, length, hidden_dim]
        assert len(unpooled.shape) == 3
        return unpooled[:, 0]
