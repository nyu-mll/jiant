from allennlp.modules.elmo_lstm import ElmoLstm


class BiLMEncoder(ElmoLstm):
    """Wrapper around BiLM to give it an interface to comply with SentEncoder
    See base class: ElmoLstm
    """

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.hidden_size * 2
