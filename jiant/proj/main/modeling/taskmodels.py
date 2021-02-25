import abc

from dataclasses import dataclass
from typing import Any
from typing import Callable

import torch
import torch.nn as nn

import jiant.proj.main.modeling.heads as heads
from jiant.proj.main.components.outputs import LogitsOutput, LogitsAndLossOutput
from jiant.utils.python.datastructures import take_one
from jiant.shared.model_resolution import ModelArchitectures
from jiant.tasks.core import TaskTypes


class JiantTaskModelFactory:
    """This factory is used to create task models bundling the task,
       encoder, and task head within the task model.

    Attributes:
        registry (dict): Dynamic registry mapping task types to task models
    """

    registry = {}

    @classmethod
    def register(cls, task_type: TaskTypes) -> Callable:
        """Register task_type as a key mapping to a TaskModel

        Args:
            task_type (TaskTypes): TaskType key mapping to a BaseHead task head

        Returns:
            Callable: inner_wrapper() wrapping TaskModel constructor
        """

        def inner_wrapper(wrapped_class: Taskmodel) -> Callable:
            assert task_type not in cls.registry
            cls.registry[task_type] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def __call__(cls, task, jiant_transformers_model, head, **kwargs):
        """This creates the TaskModel corresponding to the Task, abc.abstractmethod,
            and encoder used.

        Args:
            task (Task): Task
            jiant_transformers_model (JiantTransformersModel): Encoder
            head (BaseHead): Task head
            **kwargs: Additional arguments for initializing TaskModel

        Returns:
            TaskModel: Initialized task model bundling task, encoder, and head
        """
        taskmodel_class = cls.registry[task.TASK_TYPE]
        taskmodel = taskmodel_class(task, jiant_transformers_model, head, **kwargs)
        return taskmodel


class Taskmodel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, task, encoder, head):
        super().__init__()
        self.task = task
        self.encoder = encoder
        self.head = head

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        raise NotImplementedError


@JiantTaskModelFactory.register(TaskTypes.CLASSIFICATION)
class ClassificationModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.ClassificationHead, **kwargs):

        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):

        encoder_output = get_output_from_encoder_and_batch(encoder=self.encoder, batch=batch)
        logits = self.head(pooled=encoder_output.pooled)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.head.num_labels), batch.label_id.view(-1),)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.REGRESSION)
class RegressionModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.RegressionHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(encoder=self.encoder, batch=batch)
        # TODO: Abuse of notation - these aren't really logits  (issue #1187)
        logits = self.head(pooled=encoder_output.pooled)
        if compute_loss:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), batch.label.view(-1))
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.MULTIPLE_CHOICE)
class MultipleChoiceModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.RegressionHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)
        self.num_choices = task.NUM_CHOICES

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        input_ids = batch.input_ids
        segment_ids = batch.segment_ids
        input_mask = batch.input_mask

        choice_score_list = []
        encoder_output_other_ls = []
        for i in range(self.num_choices):
            encoder_output = get_output_from_encoder(
                encoder=self.encoder,
                input_ids=input_ids[:, i],
                segment_ids=segment_ids[:, i],
                input_mask=input_mask[:, i],
            )
            choice_score = self.head(pooled=encoder_output.pooled)
            choice_score_list.append(choice_score)
            encoder_output_other_ls.append(encoder_output.other)

        reshaped_outputs = []
        if encoder_output_other_ls[0]:
            for j in range(len(encoder_output_other_ls[0])):
                reshaped_outputs.append(
                    [
                        torch.stack([misc[j][layer_i] for misc in encoder_output_other_ls], dim=1)
                        for layer_i in range(len(encoder_output_other_ls[0][0]))
                    ]
                )
            reshaped_outputs = tuple(reshaped_outputs)

        logits = torch.cat(
            [choice_score.unsqueeze(1).squeeze(-1) for choice_score in choice_score_list], dim=1
        )

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_choices), batch.label_id.view(-1))
            return LogitsAndLossOutput(logits=logits, loss=loss, other=reshaped_outputs)
        else:
            return LogitsOutput(logits=logits, other=reshaped_outputs)


@JiantTaskModelFactory.register(TaskTypes.SPAN_COMPARISON_CLASSIFICATION)
class SpanComparisonModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.SpanComparisonHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(encoder=self.encoder, batch=batch)
        logits = self.head(unpooled=encoder_output.unpooled, spans=batch.spans)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.head.num_labels), batch.label_id.view(-1),)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.SPAN_PREDICTION)
class SpanPredictionModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.TokenClassificationHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)
        self.offset_margin = 1000
        # 1000 is a big enough number that exp(-1000) will be strict 0 in float32.
        # So that if we add 1000 to the valid dimensions in the input of softmax,
        # we can guarantee the output distribution will only be non-zero at those dimensions.

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(encoder=self.encoder, batch=batch)
        logits = self.head(unpooled=encoder_output.unpooled)
        # Ensure logits in valid range is at least self.offset_margin higher than others
        logits_offset = logits.max() - logits.min() + self.offset_margin
        logits = logits + logits_offset * batch.selection_token_mask.unsqueeze(dim=2)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.transpose(dim0=1, dim1=2).flatten(end_dim=1), batch.gt_span_idxs.flatten(),
            )
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.MULTI_LABEL_SPAN_CLASSIFICATION)
class MultiLabelSpanComparisonModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.SpanComparisonHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(encoder=self.encoder, batch=batch)
        logits = self.head(unpooled=encoder_output.unpooled, spans=batch.spans)
        if compute_loss:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.head.num_labels), batch.label_ids.float(),)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.TAGGING)
class TokenClassificationModel(Taskmodel):
    """From RobertaForTokenClassification"""

    def __init__(self, task, encoder, head: heads.TokenClassificationHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(encoder=self.encoder, batch=batch)
        logits = self.head(unpooled=encoder_output.unpooled)
        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = batch.label_mask.view(-1) == 1
            active_logits = logits.view(-1, self.head.num_labels)[active_loss]
            active_labels = batch.label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.SQUAD_STYLE_QA)
class QAModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.QAHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(encoder=self.encoder, batch=batch)
        logits = self.head(unpooled=encoder_output.unpooled)
        if compute_loss:
            loss = compute_qa_loss(
                logits=logits,
                start_positions=batch.start_position,
                end_positions=batch.end_position,
            )
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.MASKED_LANGUAGE_MODELING)
class MLMModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.BaseMLMHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)

    def forward(self, batch, tokenizer, compute_loss: bool = False):
        masked_batch = batch.get_masked(
            mlm_probability=self.task.mlm_probability,
            tokenizer=tokenizer,
            do_mask=self.task.do_mask,
        )
        encoder_output = get_output_from_encoder(
            encoder=self.encoder,
            input_ids=masked_batch.masked_input_ids,
            segment_ids=masked_batch.segment_ids,
            input_mask=masked_batch.input_mask,
        )
        logits = self.head(unpooled=encoder_output.unpooled)
        if compute_loss:
            loss = compute_mlm_loss(logits=logits, masked_lm_labels=masked_batch.masked_lm_labels)
            return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@JiantTaskModelFactory.register(TaskTypes.EMBEDDING)
class EmbeddingModel(Taskmodel):
    def __init__(self, task, encoder, head: heads.AbstractPoolerHead, **kwargs):
        super().__init__(task=task, encoder=encoder, head=head)
        self.layer = kwargs["layer"]

    def forward(self, batch, task, tokenizer, compute_loss: bool = False):
        encoder_output = get_output_from_encoder_and_batch(
            encoder_output=self.encoder, batch=batch, output_hidden_states=True
        )

        # A tuple of layers of hidden states
        hidden_states = take_one(encoder_output.other)
        layer_hidden_states = hidden_states[self.layer]

        if isinstance(self.head, heads.MeanPoolerHead):
            logits = self.head(unpooled=layer_hidden_states, input_mask=batch.input_mask)
        elif isinstance(self.head, heads.FirstPoolerHead):
            logits = self.head(layer_hidden_states)
        else:
            raise TypeError(type(self.head))

        # TODO: Abuse of notation - these aren't really logits  (issue #1187)
        if compute_loss:
            # TODO: make this optional?   (issue #1187)
            return LogitsAndLossOutput(
                logits=logits,
                loss=torch.tensor([0.0]),  # This is a horrible hack
                other=encoder_output.other,
            )
        else:
            return LogitsOutput(logits=logits, other=encoder_output.other)


@dataclass
class EncoderOutput:

    pooled: torch.Tensor
    unpooled: torch.Tensor
    other: Any = None
    # Extend later with attention, hidden_acts, etc


def get_output_from_encoder_and_batch(encoder, batch, output_hidden_states=False) -> EncoderOutput:
    """Pass batch to encoder, return encoder model output.

    Args:
        encoder: bare model outputting raw hidden-states without any specific head.
        batch: Batch object (containing token indices, token type ids, and attention mask).

    Returns:
        EncoderOutput containing pooled and unpooled model outputs as well as any other outputs.

    """
    return get_output_from_encoder(
        encoder=encoder,
        input_ids=batch.input_ids,
        segment_ids=batch.segment_ids,
        input_mask=batch.input_mask,
        output_hidden_states=output_hidden_states,
    )


def get_output_from_encoder(
    encoder, input_ids, segment_ids, input_mask, output_hidden_states=False
) -> EncoderOutput:
    """Pass inputs to encoder, return encoder output.

    Args:
        encoder: bare model outputting raw hidden-states without any specific head.
        input_ids: token indices (see huggingface.co/transformers/glossary.html#input-ids).
        segment_ids: token type ids (see huggingface.co/transformers/glossary.html#token-type-ids).
        input_mask: attention mask (see huggingface.co/transformers/glossary.html#attention-mask).

    Raises:
        RuntimeError if encoder output contains less than 2 elements.

    Returns:
        EncoderOutput containing pooled and unpooled model outputs as well as any other outputs.

    """
    model_arch = ModelArchitectures.from_encoder(encoder)
    if model_arch in [
        ModelArchitectures.BERT,
        ModelArchitectures.ROBERTA,
        ModelArchitectures.ALBERT,
        ModelArchitectures.XLM_ROBERTA,
    ]:
        pooled, unpooled, other = get_output_from_standard_transformer_models(
            encoder=encoder,
            input_ids=input_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            output_hidden_states=output_hidden_states,
        )
    elif model_arch == ModelArchitectures.ELECTRA:
        pooled, unpooled, other = get_output_from_electra(
            encoder=encoder,
            input_ids=input_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            output_hidden_states=output_hidden_states,
        )
    elif model_arch in [
        ModelArchitectures.BART,
        ModelArchitectures.MBART,
    ]:
        pooled, unpooled, other = get_output_from_bart_models(
            encoder=encoder,
            input_ids=input_ids,
            input_mask=input_mask,
            output_hidden_states=output_hidden_states,
        )
    else:
        raise KeyError(model_arch)

    # Extend later with attention, hidden_acts, etc
    if other:
        return EncoderOutput(pooled=pooled, unpooled=unpooled, other=other)
    else:
        return EncoderOutput(pooled=pooled, unpooled=unpooled)


def get_output_from_standard_transformer_models(
    encoder, input_ids, segment_ids, input_mask, output_hidden_states=False
):
    output = encoder(
        input_ids=input_ids,
        token_type_ids=segment_ids,
        attention_mask=input_mask,
        output_hidden_states=output_hidden_states,
    )
    return output.pooler_output, output.last_hidden_state, output.hidden_states


def get_output_from_bart_models(encoder, input_ids, input_mask, output_hidden_states=False):
    # BART and mBART and encoder-decoder architectures.
    # As described in the BART paper and implemented in Transformers,
    # for single input tasks, the encoder input is the sequence,
    # the decode input is 1-shifted sequence, and the resulting
    # sentence representation is the final decoder state.
    # That's what we use for `unpooled` here.
    output = encoder(
        input_ids=input_ids, attention_mask=input_mask, output_hidden_states=output_hidden_states,
    )
    dec_all = output.decoder_hidden_states
    enc_all = output.encoder_hidden_states

    unpooled = output

    hidden_states = (enc_all + dec_all,)

    bsize, slen = input_ids.shape
    batch_idx = torch.arange(bsize).to(input_ids.device)
    # Get last non-pad index
    pooled = unpooled[batch_idx, slen - input_ids.eq(encoder.config.pad_token_id).sum(1) - 1]
    return pooled, unpooled, hidden_states


def get_output_from_electra(
    encoder, input_ids, segment_ids, input_mask, output_hidden_states=False
):
    output = encoder(
        input_ids=input_ids,
        token_type_ids=segment_ids,
        attention_mask=input_mask,
        output_hidden_states=output_hidden_states,
    )
    unpooled = output.hidden_states
    pooled = unpooled[:, 0, :]
    return pooled, unpooled, output.hidden_states


def compute_mlm_loss(logits, masked_lm_labels):
    vocab_size = logits.shape[-1]
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(logits.view(-1, vocab_size), masked_lm_labels.view(-1))


def compute_qa_loss(logits, start_positions, end_positions):
    # Do we want to keep them as 1 tensor, or multiple?
    # bs x 2 x seq_len x 1

    start_logits, end_logits = logits[:, 0], logits[:, 1]
    # Taken from: RobertaForQuestionAnswering
    # If we are on multi-GPU, split add a dimension
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss
