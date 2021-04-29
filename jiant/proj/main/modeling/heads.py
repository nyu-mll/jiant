from __future__ import annotations

import abc

import torch
import torch.nn as nn

import transformers

from jiant.ext.allennlp import SelfAttentiveSpanExtractor
from jiant.shared.model_resolution import ModelArchitectures
from jiant.tasks.core import TaskTypes
from typing import Callable
from typing import List

"""
In HuggingFace/others, these heads differ slightly across different encoder models.
We're going to abstract away from that and just choose one implementation.
"""


class JiantHeadFactory:
    """This factory is used to create task-specific heads for the supported Transformer encoders.

    Attributes:
        registry (dict): Dynamic registry mapping task types to task heads
    """

    registry = {}

    @classmethod
    def register(cls, task_type_list: List[TaskTypes]) -> Callable:
        """Register each TaskType in task_type_list as a key mapping to a BaseHead task head

        Args:
            task_type_list (List[TaskType]): List of TaskTypes that are associated to a
                                             BaseHead task head

        Returns:
            Callable: inner_wrapper() wrapping task head constructor or task head factory
        """

        def inner_wrapper(wrapped_class: BaseHead) -> Callable:
            """Summary

            Args:
                wrapped_class (BaseHead): Task head class

            Returns:
                Callable: Task head constructor or factory
            """
            for task_type in task_type_list:
                assert task_type not in cls.registry
                cls.registry[task_type] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def __call__(self, task, **kwargs) -> BaseHead:
        """Summary

        Args:
            task (Task): A task head will be created based on the task type
            **kwargs: Arguments required for task head initialization

        Returns:
            BaseHead: Initialized task head
        """
        head_class = self.registry[task.TASK_TYPE]
        head = head_class(task, **kwargs)
        return head


class BaseHead(nn.Module, metaclass=abc.ABCMeta):
    """Absract class for task heads"""

    @abc.abstractmethod
    def __init__(self):
        super().__init__()


@JiantHeadFactory.register([TaskTypes.CLASSIFICATION])
class ClassificationHead(BaseHead):
    def __init__(self, task, hidden_size, hidden_dropout_prob, **kwargs):
        """From RobertaClassificationHead"""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, len(task.LABELS))
        self.num_labels = len(task.LABELS)

    def forward(self, pooled):
        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


@JiantHeadFactory.register([TaskTypes.REGRESSION, TaskTypes.MULTIPLE_CHOICE])
class RegressionHead(BaseHead):
    def __init__(self, task, hidden_size, hidden_dropout_prob, **kwargs):
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


@JiantHeadFactory.register([TaskTypes.SPAN_COMPARISON_CLASSIFICATION])
class SpanComparisonHead(BaseHead):
    def __init__(self, task, hidden_size, hidden_dropout_prob, **kwargs):
        """From RobertaForSpanComparisonClassification"""
        super().__init__()
        self.num_spans = task.num_spans
        self.num_labels = len(task.LABELS)
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


@JiantHeadFactory.register([TaskTypes.TAGGING])
class TokenClassificationHead(BaseHead):
    def __init__(self, task, hidden_size, hidden_dropout_prob, **kwargs):
        """From RobertaForTokenClassification"""
        super().__init__()
        self.num_labels = len(task.LABELS)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, unpooled):
        unpooled = self.dropout(unpooled)
        logits = self.classifier(unpooled)
        return logits


@JiantHeadFactory.register([TaskTypes.SQUAD_STYLE_QA])
class QAHead(BaseHead):
    def __init__(self, task, hidden_size, **kwargs):
        """From RobertaForQuestionAnswering"""
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, unpooled):
        logits = self.qa_outputs(unpooled)
        # bs x seq_len x 2
        logits = logits.permute(0, 2, 1)
        # bs x 2 x seq_len x 1
        return logits


@JiantHeadFactory.register([TaskTypes.MASKED_LANGUAGE_MODELING])
class JiantMLMHeadFactory:
    """This factory is used to create masked language modeling (MLM) task heads.
    This is required due to Transformers implementing different MLM heads for
    different encoders.

    Attributes:
        registry (dict): Dynamic registry mapping model architectures to MLM task heads
    """

    registry = {}

    @classmethod
    def register(cls, model_arch_list: List[ModelArchitectures]) -> Callable:
        """Registers the ModelArchitectures in model_arch_list as keys mapping to a MLMHead

        Args:
            model_arch_list (List[ModelArchitectures]): List of ModelArchitectures mapping to
                                                        an MLM task head.

        Returns:
            Callable: MLMHead class
        """

        def inner_wrapper(wrapped_class: BaseMLMHead) -> Callable:
            for model_arch in model_arch_list:
                assert model_arch not in cls.registry
                cls.registry[model_arch] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def __call__(self, task, **kwargs):
        """Summary

        Args:
            task (Task): Task used to initialize task head
            **kwargs: Additional arguments required to initialize task head
        """
        mlm_head_class = self.registry[task.TASK_TYPE]
        mlm_head = mlm_head_class(task, **kwargs)
        return mlm_head


class BaseMLMHead(BaseHead, metaclass=abc.ABCMeta):
    pass


@JiantMLMHeadFactory.register([ModelArchitectures.BERT])
class BertMLMHead(BaseMLMHead):
    """From BertOnlyMLMHead, BertLMPredictionHead, BertPredictionHeadTransform"""

    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12, hidden_act="gelu"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transformers.models.bert.modeling_bert.ACT2FN[hidden_act]
        self.LayerNorm = transformers.models.bert.modeling_bert.BertLayerNorm(
            hidden_size, eps=layer_norm_eps
        )

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


@JiantMLMHeadFactory.register([ModelArchitectures.ROBERTA, ModelArchitectures.XLM_ROBERTA])
class RobertaMLMHead(BaseMLMHead):
    """From RobertaLMHead"""

    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = transformers.models.bert.modeling_bert.BertLayerNorm(
            hidden_size, eps=layer_norm_eps
        )

        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)

        # Need a link between the two variables so that the bias is correctly resized with
        # `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, unpooled):
        x = self.dense(unpooled)
        x = transformers.models.bert.modeling_bert.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        logits = self.decoder(x) + self.bias
        return logits


@JiantMLMHeadFactory.register([ModelArchitectures.ALBERT])
class AlbertMLMHead(BaseMLMHead):
    """From AlbertMLMHead"""

    def __init__(self, hidden_size, embedding_size, vocab_size, hidden_act="gelu"):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(embedding_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, vocab_size)
        self.activation = transformers.models.bert.modeling_bert.ACT2FN[hidden_act]

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
