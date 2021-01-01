import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import copy
import math

import jiant.shared.task_aware_unit as tau


class TransNorm(
    tau.TauMixin, nn.Module,
):
    def __init__(self, layer_norm_layer, task_names, momentum, transnorm_skip):
        super().__init__()
        self.normalized_shape = layer_norm_layer.normalized_shape
        self.eps = layer_norm_layer.eps
        self.elementwise_affine = layer_norm_layer.elementwise_affine
        self.task_names = task_names
        self.transnorm_skip = transnorm_skip
        if self.elementwise_affine:
            self.weight = layer_norm_layer.weight
            self.bias = layer_norm_layer.bias
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        for task_name in task_names:
            self.register_buffer(f"{task_name}_mean", torch.zeros(self.normalized_shape))
            self.register_buffer(f"{task_name}_var", torch.ones(self.normalized_shape))
        self.momentum = momentum

    def forward(self, input):
        if self.training:
            with torch.no_grad():
                reshaped_input = input.flatten(end_dim=-len(self.normalized_shape) - 1)
                self._buffers[f"{self.tau_task_name}_mean"] *= 1 - self.momentum
                self._buffers[f"{self.tau_task_name}_mean"] += self.momentum * reshaped_input.mean(
                    dim=0
                )
                self._buffers[f"{self.tau_task_name}_var"] *= 1 - self.momentum
                self._buffers[f"{self.tau_task_name}_var"] += self.momentum * reshaped_input.var(
                    dim=0
                )
        layer_norm_output = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
        discrepency = torch.stack(
            [
                self._buffers[f"{task_name}_mean"]
                / torch.sqrt(self._buffers[f"{task_name}_var"] + self.eps)
                for task_name in self.task_names
            ],
            dim=0,
        )
        alpha = 1 / (1 + discrepency.max(dim=0)[0] - discrepency.min(dim=0)[0])
        alpha = alpha / torch.sum(alpha) * math.prod(self.normalized_shape)
        if self.transnorm_skip:
            alpha = 1 + alpha
        target_shape = [1] * (len(layer_norm_output.size()) - len(alpha.size())) + list(
            alpha.size()
        )
        output = layer_norm_output * alpha.view(*target_shape)
        return output


def replace_layernorm_with_transnorm(
    encoder, num_layers, task_names, transnorm_update_rate, transnorm_skip
):
    for idx in range(num_layers):
        encoder.layer[idx].attention.output.LayerNorm = TransNorm(
            layer_norm_layer=encoder.layer[idx].attention.output.LayerNorm,
            task_names=task_names,
            momentum=transnorm_update_rate,
            transnorm_skip=transnorm_skip,
        )
        encoder.layer[idx].output.LayerNorm = TransNorm(
            layer_norm_layer=encoder.layer[idx].output.LayerNorm,
            task_names=task_names,
            momentum=transnorm_update_rate,
            transnorm_skip=transnorm_skip,
        )


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * F.sigmoid(input)


class BertOutputWithAdapter(tau.TauMixin, nn.Module):
    def __init__(
        self, bert_output_layer, task_names, hidden_size, reduction_factor=16, non_linearity="relu",
    ):
        super().__init__()
        self.dense = bert_output_layer.dense
        self.LayerNorm = bert_output_layer.LayerNorm
        self.dropout = bert_output_layer.dropout
        if non_linearity == "relu":
            non_linear_module = nn.ReLU()
        elif non_linearity == "leaky_relu":
            non_linear_module = nn.LeakyReLU()
        elif non_linearity == "swish":
            non_linear_module = Swish()
        self.adapters = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // reduction_factor),
                    non_linear_module,
                    nn.Linear(hidden_size // reduction_factor, hidden_size),
                )
                for task_name in task_names
            }
        )
        for a_module in self.adapters.modules():
            if isinstance(a_module, nn.Linear):
                a_module.weight.data.normal_(mean=0.0, std=0.02)
                a_module.bias.data.zero_()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        adapter_inputs = self.LayerNorm(input_tensor + hidden_states)
        adapter_states = self.adapters[self.tau_task_name](adapter_inputs) + hidden_states
        hidden_states = self.LayerNorm(input_tensor + adapter_states)
        return hidden_states


class BertOutputWithAdapterFusion(BertOutputWithAdapter):
    def __init__(
        self, bert_output_layer, task_names, hidden_size, reduction_factor=16, non_linearity="relu",
    ):
        super().__init__(
            bert_output_layer, task_names, hidden_size, reduction_factor, non_linearity
        )

        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        with torch.no_grad():
            self.key_layer.weight.data.normal_(0, 0.02)
            self.query_layer.weight.data.normal_(0, 0.02)
            self.value_layer.weight.copy_(
                torch.zeros(int(hidden_size), int(hidden_size)) + 0.000001
            ).fill_diagonal_(1.0)

    def forward(self, hidden_states, input_tensor, attention_mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        prenorm_adapter_inputs = input_tensor + hidden_states
        adapter_inputs = self.LayerNorm(prenorm_adapter_inputs)
        adapter_outputs = torch.cat(
            [adapter(adapter_inputs) for adapter in self.adapters.values()], dim=0
        )
        key = self.key_layer(adapter_outputs)
        value = self.value_layer(adapter_outputs + hidden_states)
        query = self.query_layer(prenorm_adapter_inputs)
        attention_scores = self.dropout(
            torch.sum(query.unsqueeze(dim=0) * key, dim=-1, keepdim=True)
        )
        attention_probs = F.softmax(attention_scores, dim=0)
        fusion_outputs = torch.sum(attention_probs * value, dim=0)
        hidden_states = self.LayerNorm(input_tensor + fusion_outputs)
        return hidden_states


class SluiceEncoder(tau.TauMixin, nn.Module):
    def __init__(
        self,
        bert_encoder,
        bert_config,
        task_a,
        task_b,
        sluice_num_subspaces,
        sluice_init_var,
        sluice_lr_multiplier,
    ):
        super().__init__()
        self.layer_a = bert_encoder.layer
        self.layer_b = copy.deepcopy(self.layer_a)
        self.task_sharing = nn.ModuleList(
            [
                SluiceTaskSharingUnit(
                    bert_config.hidden_size,
                    sluice_num_subspaces,
                    sluice_init_var,
                    sluice_lr_multiplier,
                )
                for i in range(bert_config.num_hidden_layers)
            ]
        )
        self.layer_sharing_a = SluiceLayerSharingUnit(
            bert_config.num_hidden_layers,
            bert_config.hidden_size,
            sluice_num_subspaces,
            sluice_lr_multiplier,
        )
        self.layer_sharing_b = SluiceLayerSharingUnit(
            bert_config.num_hidden_layers,
            bert_config.hidden_size,
            sluice_num_subspaces,
            sluice_lr_multiplier,
        )
        self.task_a = task_a
        self.task_b = task_b

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        assert not output_attentions
        hidden_states_a = hidden_states_b = hidden_states
        all_hidden_states = (hidden_states,)
        for i, (layer_module_a, layer_module_b) in enumerate(zip(self.layer_a, self.layer_b)):

            layer_outputs_a = layer_module_a(
                hidden_states_a,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
            layer_outputs_b = layer_module_b(
                hidden_states_b,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states_a, hidden_states_b = layer_outputs_a[0], layer_outputs_b[0]

            if self.tau_task_name == self.task_a:
                all_hidden_states = all_hidden_states + (hidden_states_a,)
            elif self.tau_task_name == self.task_b:
                all_hidden_states = all_hidden_states + (hidden_states_b,)
            else:
                assert False

            hidden_states_a, hidden_states_b = self.task_sharing[i](
                hidden_states_a, hidden_states_b
            )

        if self.tau_task_name == self.task_a:
            hidden_states = self.layer_sharing_a(all_hidden_states)
        elif self.tau_task_name == self.task_b:
            hidden_states = self.layer_sharing_b(all_hidden_states)
        else:
            assert False

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return transformers.modeling_bert.BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states,
        )


class SluiceTaskSharingUnit(nn.Module):
    def __init__(self, hidden_size, sluice_num_subspaces, sluice_init_var, sluice_lr_multiplier):
        super().__init__()
        self.subspace_size = hidden_size // sluice_num_subspaces
        self.sluice_lr_multiplier = sluice_lr_multiplier
        self.exchange_matrix = nn.Parameter(torch.eye(sluice_num_subspaces * 2))
        self.exchange_matrix.data.normal_(0, sluice_init_var).fill_diagonal_(1)
        self.exchange_matrix.data /= self.sluice_lr_multiplier
        self.register_buffer(
            "channel_mask",
            torch.eye(self.subspace_size).repeat(
                2 * sluice_num_subspaces, 2 * sluice_num_subspaces
            ),
        )

    def forward(self, input_a, input_b):
        full_exchange_matrix = (self.exchange_matrix * self.sluice_lr_multiplier).repeat_interleave(
            self.subspace_size, dim=0
        ).repeat_interleave(self.subspace_size, dim=1) * self.channel_mask
        input_ab = torch.cat([input_a, input_b], dim=-1)
        output_ab = torch.matmul(input_ab, full_exchange_matrix)
        output_a, output_b = torch.chunk(output_ab, chunks=2, dim=-1)
        return output_a, output_b


class SluiceLayerSharingUnit(nn.Module):
    def __init__(self, num_layers, hidden_size, sluice_num_subspaces, sluice_lr_multiplier):
        super().__init__()
        self.subspace_size = hidden_size // sluice_num_subspaces
        self.sluice_lr_multiplier = sluice_lr_multiplier
        self.mix_matrix = nn.Parameter(torch.zeros(num_layers + 1, sluice_num_subspaces))
        self.mix_matrix.data[-1] += 1
        self.mix_matrix.data /= self.sluice_lr_multiplier

    def forward(self, all_layer_states):
        full_mix_matrix = (self.mix_matrix * self.sluice_lr_multiplier).repeat_interleave(
            self.subspace_size, dim=1
        )
        output = torch.sum(torch.stack(all_layer_states, dim=-2) * full_mix_matrix, dim=-2)
        return output
