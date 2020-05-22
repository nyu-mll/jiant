import contextlib
import transformers


@contextlib.contextmanager
def output_hidden_states_context(encoder):
    assert isinstance(
        encoder,
        (
            transformers.BertModel,
            transformers.RobertaModel,
            transformers.AlbertModel,
            transformers.XLMRobertaModel,
        ),
    )
    assert hasattr(encoder.encoder, "output_hidden_states")
    old_value = encoder.encoder.output_hidden_states
    encoder.encoder.output_hidden_states = True
    yield
    encoder.encoder.output_hidden_states = old_value
