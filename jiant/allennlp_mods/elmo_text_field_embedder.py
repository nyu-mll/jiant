# Modified from allennlp/allennlp/modules/text_field_embedders/basic_text_field_embedder.py
# This textfield embedder is compatible with multi- output representation ELMo. In the Basic
# version, the single ELMo output representation can be simply concatenated. In multi-output
# representation, we need to first select the representation that is used.

# A wrapper class for Elmo is also included here.

from typing import Dict

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Elmo
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from overrides import overrides


@TokenEmbedder.register("elmo_token_embedder_wrapper")
class ElmoTokenEmbedderWrapper(TokenEmbedder):
    """
    Wraps the Elmo call so that the parameters are saved correctly

    Forwards all calls to Elmo
    """

    def __init__(
        self,
        options_file: str,
        weight_file: str,
        do_layer_norm: bool = False,
        dropout: float = 0.5,
        requires_grad: bool = False,
        projection_dim: int = None,
        num_output_representations: int = 1,
    ) -> None:
        super(ElmoTokenEmbedderWrapper, self).__init__()

        # other arguments can be passed in when needed
        self._elmo = Elmo(
            options_file=options_file,
            weight_file=weight_file,
            num_output_representations=num_output_representations,
            dropout=dropout,
        )

    def get_output_dim(self):
        return self._elmo.get_output_dim()

    def forward(
        self, inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        return self._elmo(inputs)

    # this is also deferred to elmo
    @classmethod
    def from_params(cls, params: Params):
        self._elmo = Elmo.from_params(params)
        return self


@TextFieldEmbedder.register("elmo")
class ElmoTextFieldEmbedder(TextFieldEmbedder):
    """
    forward() now accepts classifier name as an argument, which tells the embedder which ELMo representation
    to return. init() also requires a dict of classifier names (i.e. the number of tasks that need their own
    ELMo scalars). which map to an int corresponding to their elmo scalars in the elmo object. These are
    names (strings) and not necessarily the same as task names (e.g. mnli for mnli-diagnostic).

    This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.  Each
    ``TokenEmbedder`` embeds or encodes the representation output from one
    :class:`~allennlp.data.TokenIndexer`.  As the data produced by a
    :class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
    representations, we take ``TokenEmbedders`` with corresponding names.  Each ``TokenEmbedders``
    embeds its input, and the result is concatenated in an arbitrary order.
    """

    def __init__(
        self,
        token_embedders: Dict[str, TokenEmbedder],
        classifiers: Dict[str, int],
        elmo_chars_only=False,  # Flag ensuring we are using real ELMo
        sep_embs_for_skip=False,
    ) -> None:  # Flag indicating separate scalars per task
        super(ElmoTextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        for key, embedder in token_embedders.items():
            name = "token_embedder_%s" % key
            self.add_module(name, embedder)
        self.task_map = classifiers  # map handling classifier_name -> scalar idx
        self.elmo_chars_only = elmo_chars_only
        self.sep_embs_for_skip = sep_embs_for_skip

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(
        self,
        text_field_input: Dict[str, torch.Tensor],
        classifier_name: str = "@pretrain@",
        num_wrapping_dims: int = 0,
    ) -> torch.Tensor:
        if self._token_embedders.keys() != text_field_input.keys():
            message = "Mismatched token keys: %s and %s" % (
                str(self._token_embedders.keys()),
                str(text_field_input.keys()),
            )
            raise ConfigurationError(message)
        embedded_representations = []
        keys = sorted(text_field_input.keys())
        for key in keys:
            tensor = text_field_input[key]
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, "token_embedder_{}".format(key))
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            token_vectors = embedder(tensor)

            # Changed vs original:
            # If we want separate scalars/task, figure out which representation to use, since
            # embedder create a representation for _all_ sets of scalars. This can be optimized
            # with more wrapper classes but we compute all of them for now.
            # The shared ELMo scalar weights version all use the @pretrain@ embeddings.
            # There must be at least as many ELMo representations as the highest index in
            # self.task_map, otherwise indexing will fail.
            if key == "elmo" and not self.elmo_chars_only:
                if self.sep_embs_for_skip:
                    token_vectors = token_vectors["elmo_representations"][
                        self.task_map[classifier_name]
                    ]
                else:
                    token_vectors = token_vectors["elmo_representations"][
                        self.task_map["@pretrain@"]
                    ]

            # optional projection step that we are ignoring.
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> "BasicTextFieldEmbedder":
        token_embedders = {}
        keys = list(params.keys())
        for key in keys:
            embedder_params = params.pop(key)
            token_embedders[key] = TokenEmbedder.from_params(vocab, embedder_params)
        params.assert_empty(cls.__name__)
        return cls(token_embedders)
