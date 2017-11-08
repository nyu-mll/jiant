import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import Module, LSTMCell, Linear, Parameter, Softmax
from abc import ABCMeta, abstractmethod, abstractproperty

class RNNDecoder(Module):
    def __init__(self, decoder_cell, token_embedder, rnn_context_combiner):
        """Construct RNNDecoder.

        Args:
            decoder_cell (DecoderCell)
            token_embedder (TokenEmbedder)
            rnn_context_combiner (RNNContextCombiner)
        """
        super(RNNDecoder, self).__init__()
        self.decoder_cell = decoder_cell
        self.token_embedder = token_embedder
        self.word_vocab = token_embedder.vocab
        self.rnn_context_combiner = rnn_context_combiner

    def forward(self, encoder_output, train_decoder_input):
        """

        Args:
            encoder_output (EncoderOutput)
            train_decoder_input (TrainDecoderInput)

        Returns:
            rnn_states (list[RNNState])
            total_loss (Variable): a scalar loss
        """
        batch_size, _ = train_decoder_input.input_words.mask.size()
        rnn_state = self.decoder_cell.initialize(batch_size)

        input_word_embeds = self.token_embedder.embed_seq_batch(train_decoder_input.input_words)

        input_embed_list = input_word_embeds.split()
        target_word_list = train_decoder_input.target_words.split()

        loss_list = []
        rnn_states = []
        for t, (x, target_word) in enumerate(izip(input_embed_list, target_word_list)):
            # x is a (batch_size, word_dim) SequenceBatchElement, target_word is a (batch_size,) Variable

            # update rnn state
            rnn_input = self.rnn_context_combiner(encoder_output, x.values)
            decoder_cell_output = self.decoder_cell(rnn_state, rnn_input, x.mask)
            rnn_state = decoder_cell_output.rnn_state
            rnn_states.append(rnn_state)

            # compute loss
            loss = decoder_cell_output.loss(target_word.values)  # (batch_size,)
            loss_list.append(SequenceBatchElement(loss, x.mask))

        losses = SequenceBatch.cat(loss_list)  # (batch_size, target_seq_length)

        # sum losses across time, accounting for mask
        per_instance_losses = SequenceBatch.reduce_sum(losses)  # (batch_size,)
        return rnn_states, per_instance_losses

    def rnn_states(self, encoder_output, train_decoder_input):
        rnn_states, _ = self(encoder_output, train_decoder_input)
        return rnn_states

    def per_instance_losses(self, encoder_output, train_decoder_input):
        _, per_instance_losses = self(encoder_output, train_decoder_input)
        return per_instance_losses

    def loss(self, encoder_output, train_decoder_input):
        _, per_instance_losses = self(encoder_output, train_decoder_input)
        total_loss = torch.mean(per_instance_losses)
        return total_loss

class DecoderCell(Module):
    __metaclass__ = ABCMeta

    @abstractproperty
    def rnn_state_type(self):
        pass

    @abstractproperty
    def rnn_input_type(self):
        pass

    @abstractmethod
    def initialize(self, batch_size):
        """Return initial RNNState.

        Args:
            batch_size (int)

        Returns:
            RNNState
        """
        raise NotImplementedError

    def forward(self, rnn_state, rnn_input, advance):
        """Advance the decoder by one step.

        Args:
            rnn_state (RNNState): the previous RNN state.
            rnn_input (RNNInput): any inputs at this time step.
            advance (Variable): of shape (batch_size, 1). The RNN should advance on example i iff mask[i] == 1.

        Returns:
            DecoderCellOutput
        """
        raise NotImplementedError

class SimpleDecoderCell(DecoderCell):
    def __init__(self, token_embedder, hidden_dim, input_dim, agenda_dim):
        super(SimpleDecoderCell, self).__init__()
        self.rnn_cell = LSTMCell(input_dim + agenda_dim, hidden_dim)
        self.linear = Linear(hidden_dim, input_dim)
        self.h0 = Parameter(torch.zeros(hidden_dim))
        self.c0 = Parameter(torch.zeros(hidden_dim))
        self.softmax = Softmax()
        self.token_embedder = token_embedder

    @property
    def rnn_state_type(self):
        return SimpleRNNState

    @property
    def rnn_input_type(self):
        return SimpleRNNInput

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)
        return SimpleRNNState(h, c)

    def forward(self, rnn_state, rnn_input, advance):
        rnn_input_embed = torch.cat([rnn_input.x, rnn_input.agenda], 1)
        h, c = self.rnn_cell(rnn_input_embed, (rnn_state.h, rnn_state.c))

        # don't update if sequence has terminated
        h = gated_update(rnn_state.h, h, advance)
        c = gated_update(rnn_state.c, c, advance)

        query = self.linear(h)
        word_vocab = self.token_embedder.vocab
        word_embeds = self.token_embedder.embeds
        vocab_logits = torch.mm(query, word_embeds.t())  # (batch_size, vocab_size)
        vocab_probs = self.softmax(vocab_logits)

        # no attention over source, insert and delete embeds
        rnn_state = SimpleRNNState(h, c)

        return DecoderCellOutput(rnn_state, vocab=word_vocab, vocab_probs=vocab_probs)

class MultilayeredDecoderCell(DecoderCell):
    def __init__(self, token_embedder, hidden_dim, input_dim, agenda_dim, num_layers):
        super(MultilayeredDecoderCell, self).__init__()
        self.linear = Linear(hidden_dim, input_dim)
        self.h0 = Parameter(torch.zeros(hidden_dim))
        self.c0 = Parameter(torch.zeros(hidden_dim))
        self.softmax = Softmax()
        self.token_embedder = token_embedder
        self.num_layers = num_layers

        self.rnn_cells = []
        for layer in range(num_layers):
            in_dim = (input_dim + agenda_dim) if layer == 0 else hidden_dim # inputs to first layer are word vectors
            out_dim = hidden_dim
            rnn_cell = LSTMCell(in_dim, out_dim)
            self.add_module('decoder_layer_{}'.format(layer), rnn_cell)
            self.rnn_cells.append(rnn_cell)

    @property
    def rnn_state_type(self):
        return MultilayeredRNNState

    @property
    def rnn_input_type(self):
        return MultilayeredRNNInput

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)
        return MultilayeredRNNState([h] * self.num_layers, [c] * self.num_layers)

    def forward(self, rnn_state, rnn_input, advance):
        x = torch.cat([rnn_input.x, rnn_input.agenda], 1)
        hs, cs = [], []
        for layer in range(self.num_layers):
            rnn_cell = self.rnn_cells[layer]

            # collect the h, c belonging to the previous time-step at the corresponding depth
            h_prev_t, c_prev_t = rnn_state.hs[layer], rnn_state.cs[layer]

            # forward pass and masking
            h, c = rnn_cell(x, (h_prev_t, c_prev_t))
            h = gated_update(h_prev_t, h, advance)
            c = gated_update(c_prev_t, c, advance)
            hs.append(h)
            cs.append(c)

            if layer == 0:
                x = h  # no skip connection on the first layer
            else:
                x = x + h

        query = self.linear(x)
        word_vocab = self.token_embedder.vocab
        word_embeds = self.token_embedder.embeds
        vocab_logits = torch.mm(query, word_embeds.t())  # (batch_size, vocab_size)
        vocab_probs = self.softmax(vocab_logits)

        rnn_state = MultilayeredRNNState(hs, cs)

        return DecoderCellOutput(rnn_state, vocab=word_vocab, vocab_probs=vocab_probs)

class SimpleRNNDecoder(nn.Module):
    '''
    Latter half of the decoder in the CVAE. Takes the encoded prototype
        and the edit vector and decodes from them.
    Will implement as an LSTM with attention over encoder states and
        edit vector concatenated to input at each step.
    '''

    def __init__(self, input_dim, hid_dim, embs):
        '''

        Args:
            input_dim (int): dimension of input at each time step
            hid_dim (int)
            embs (Variable): output elements to compute prob dist over, size
                             (out_dim, hid_dim)
        '''
        super(SimpleRNNDecoder, self).__init__()

        self.rnn_cell = LSTMCell(input_dim, hid_dim) # GRU?
        self.embs = embs
        out_dim = embs.size()[-1]
        self.h0 = Parameter(torch.zeros(hid_dim))
        self.c0 = Parameter(torch.zeros(hid_dim))
        self.linear = Linear(hid_dim, out_dim)
        self.softmax = Softmax()

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)
        return h, c

    def forward(self, inputs, targs):
        '''

        Args:
            inputs (list[Variable]): list of length seq_len of Variables of 
                                size (batch_size, input_dim)
            targs (list[Variable]): list of length seq_len of Variables of
                                size (batch_size, ) containing idxs of targ

        Returns:
            rnn_states (list[(Variable, Variable)]): list of tuples of hidden 
                                                     and cell states
            total_loss (Variable): losses per batch example, size (batch_size,)
        '''
        pdb.set_trace()
        batch_size = inputs[0].size()[0]
        h, c = self.rnn_cell.initialize(batch_size)

        rnn_states, losses = [], []
        for t, (inp, targ) in enumerate(zip(inputs, targs)):
            # inp is (batch_size, dim); targ is (batch_size, ) of target idx

            h_t, c_t = self.rnn_cell(inp, (h, c))
            h = h_t #gated_update(rnn_state.h, h, mask)
            c = c_t #gated_update(rnn_state.c, c, mask)

            query = self.linear(h)
            embs = self.embs
            logits = torch.mm(query, embs.t())
            probs = self.softmax(logits)
            rnn_states.append((h, c))

            targ_probs = torch.gather(probs, 1, targ.unsqueeze(1)).squeeze(1)
            loss = -torch.log(targ_probs + 1e-45) # NLL
            losses.append(loss)

        losses = torch.cat([loss.unsqueeze(1) for loss in losses])
        instance_losses = losses.reduce_sum(losses, dim=1).squeeze(dim=1)
        return rnn_states, instance_losses

    def states(self, inputs, targs):
        rnn_states, _ = self(inputs, targs)
        return rnn_states

    def instance_losses(self, inputs, targs):
        _, instance_losses = self(inputs, targs)
        return instance_losses

    def loss(self, inputs, targs):
        _, instance_losses = self(inputs, targs)
        total_loss = torch.mean(instance_losses)
        return total_loss



class MultilayerRNNDecoder(nn.Module):
    '''
    '''

    def __init__(self, input_dim, hid_dim, n_layers, embs):
        '''

        Args:
            input_dim (int): dimension of input to first layer
            hid_dim (int): dimension of hidden state
            n_layers (int)
            embs (Variable): output elements to compute prob dist over, size
                             (out_dim, hid_dim)
        '''
        super(MultilayerRNNDecoder, self).__init__()

        self.embs = embs
        out_dim = embs.size()[-1]
        self.h0 = Parameter(torch.zeros(hid_dim))
        self.c0 = Parameter(torch.zeros(hid_dim))
        self.linear = Linear(hid_dim, out_dim)
        self.softmax = Softmax()
        self.n_layers = n_layers
        self.rnn_cells = []
        for layer in range(n_layers):
            in_dim = hid_dim if layer else input_dim
            rnn_cell = LSTMCell(in_dim, hid_dim)
            self.layers.append(rnn_cell)
            self.add_module('decoder_layer_{}'.format(layer), rnn_cell)

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)
        return [h] * self.n_layers, [c] * self.n_layers # are these tied?

    def forward(self, inputs, targs):
        '''

        Args:
            inputs (list[Variable]): list of length seq_len of Variables of 
                                size (batch_size, input_dim)
            targs (list[Variable]): list of length seq_len of Variables of
                                size (batch_size, ) containing idxs of targ

        Returns:
            rnn_states (list[(Variable, Variable)]): list of tuples of hidden 
                                                     and cell states
            total_loss (Variable): losses per batch example, size (batch_size,)
        '''
        batch_size, _ = inputs.size()
        old_hs, old_cs = self.rnn_cell.initialize(batch_size)

        rnn_states, losses = [], []
        for t, (inp, targ) in enumerate(zip(inputs, targs)):
            # inp is (batch_size, dim); targ is (batch_size, ) of target idx

            new_hs, new_cs = [], []
            for layer in range(self.n_layers):
                rnn_cell = self.rnn_cells[layer]
                old_h_t, old_c__t = old_hs[layer], old_cs[layer]
                h_t, c_t = rnn_cell(inp, (old_h_t, old_c_t))
                h = h_t #gated_update(rnn_state.h, h, mask)
                c = c_t #gated_update(rnn_state.c, c, mask)
                new_hs.append(h)
                new_cs.append(c)

                if layer:
                    inp = inp + h # skip connections
                else:
                    inp = h # except on first layer
            old_hs, old_cs = new_hs, new_cs
            rnn_states.append((old_hs, old_cs))

            query = self.linear(inp)
            embs = self.embs
            logits = torch.mm(query, embs.t())
            probs = self.softmax(logits)
            targ_probs = torch.gather(probs, 1, targ.unsqueeze(1)).squeeze(1)
            loss = -torch.log(targ_probs + 1e-45) # NLL
            losses.append(loss)

        losses = torch.cat([loss.unsqueeze(1) for loss in losses])
        instance_losses = losses.reduce_sum(losses, dim=1).squeeze(dim=1)
        return rnn_states, instance_losses

    def states(self, inputs, targs):
        rnn_states, _ = self(inputs, targs)
        return rnn_states

    def instance_losses(self, inputs, targs):
        _, instance_losses = self(inputs, targs)
        return instance_losses

    def loss(self, inputs, targs):
        _, instance_losses = self(inputs, targs)
        total_loss = torch.mean(instance_losses)
        return total_loss



class AttentionRNNDecoder(nn.Module):
    '''
    Attention RNN cell
    '''
    
    def __init__(self, word_dim, agenda_dim, decoder_dim, encoder_dim, \
                 attn_dim, no_insert_delete_attn, n_layers):
        '''
        Multi-layer RNN decoder with attention using top layer hidden states

        Args:
            - input_dim (int)
            - n_layers
        '''
        super(AttentionRNNDecoder, self).__init__()

        self.n_layers = n_layers

        # augment the input to each RNN layer with the edit vec and attn over
        #   - encoder states
        #   - insert words, optionally
        #   - delete words, optionally
        augment_dim = encoder_dim + word_dim + word_dim + agenda_dim

        self.rnn_cells = []
        for layer in range(num_layers):
            in_dim = word_dim if layer == 0 else decoder_dim
            out_dim = decoder_dim
            rnn_cell = LSTMCell(in_dim + augment_dim, out_dim)
            self.add_module('decoder_layer_{}'.format(layer), rnn_cell)
            self.rnn_cells.append(rnn_cell)

        # see definition of `z` in `forward` method
        # to predict words, we condition on the hidden state h + 3 attention contexts
        z_dim = decoder_dim + encoder_dim + 2 * word_dim
        if no_insert_delete_attn:
            z_dim = decoder_dim + encoder_dim

        # TODO(Alex): these big params may need regularization
        self.vocab_projection_pos = Linear(z_dim, word_dim)
        self.vocab_projection_neg = Linear(z_dim, word_dim)
        self.relu = torch.nn.ReLU()

        self.h0 = Parameter(torch.zeros(decoder_dim))
        self.c0 = Parameter(torch.zeros(decoder_dim))
        self.softmax = Softmax()

        self.source_attn = Attention(encoder_dim, decoder_dim, attn_dim)
        if not no_insert_delete_attn:
            self.insert_attn = Attention(input_dim, decoder_dim, attn_dim)
            self.delete_attn = Attention(input_dim, decoder_dim, attn_dim)
        else:
            self.insert_attn = DummyAttention(input_dim, decoder_dim, attn_dim)
            self.delete_attn = DummyAttention(input_dim, decoder_dim, attn_dim)

        self.token_embedder = token_embedder
        self.no_insert_delete_attn = no_insert_delete_attn

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)

        # no initial weights, context is just zero vector
        init_attn = lambda attn: torch.zeros(batch_size, attn.memory_dim)

        return ([h] * self.num_layers, [c] * self.num_layers, \
               init_attn(self.source_attn), init_attn(self.insert_attn), \
               init_attn(self.delete_attn))

    def forward(self, inputs, targs, agenda):
        batch_size, _ = inputs.size()
        old_hs, old_cs, src_attn, insert_attn, delete_attn = \
                self.initialize(batch_size)

        rnn_states, losses = [], []
        for t, (inp, targ) in enumerate(zip(inputs, targs)):
            # inp is (batch_size, dim); targ is (batch_size, ) of target idx

            augmentation = torch.cat([source_attn, insert_attn, delete_attn, 
                                      agenda], 1)

            new_hs, new_cs = [], []
            for layer in range(self.num_layers):
                rnn_cell = self.rnn_cells[layer]
                old_h, old_c = old_hs[layer], old_cs[layer]
                rnn_input = torch.cat([inp, augmentation], 1)
                h, c = rnn_cell(rnn_input, (old_h, old_c))
                h = h #gated_update(old_h, h, mask)
                c = c #gated_update(old_c, c, mask)
                new_hs.append(h)
                new_cs.append(c)

                if layer == 0:
                    inp = h  # no skip connection on the first layer
                else:
                    inp = inp + h
            old_hs, old_cs = new_hs, new_cs
            rnn_states.append((old_hs, old_cs))

            # compute attention using bottom layer; TODO pass *_embeds
            source_attn = self.source_attn(dci.source_embeds, old_hs[0])
            insert_attn = self.insert_attn(dci.insert_embeds, old_hs[0])
            delete_attn = self.delete_attn(dci.delete_embeds, old_hs[0])
            if not self.no_insert_delete_attn:
                z = torch.cat([x, source_attn, insert_attn, delete_attn], 1)
            else:
                z = torch.cat([x, source_attn], 1)
            # has shape (batch_size, decoder_dim + encoder_dim + input_dim + input_dim)

            query_pos = self.projection_pos(z)
            query_neg = self.projection_neg(z)
            word_embeds = self.embs
            # (batch_size, vocab_size)
            logit_pos = self.relu(torch.mm(query_pos, embs.t())) 
            logit_neg = self.relu(torch.mm(query_neg, embs.t()))
            # TODO(Alex?): prevent model from putting probability on UNK
            probs = self.softmax(logit_pos - logit_neg)
            targ_probs = torch.gather(probs, 1, targ.unsqueeze(1)).squeeze(1)
            loss = -torch.log(targ_probs + 1e-45) # NLL
            losses.append(loss)

        losses = torch.cat([loss.unsqueeze(1) for loss in losses])
        instance_losses = losses.reduce_sum(losses, dim=1).squeeze(dim=1)
        return rnn_states, instance_losses

    def states(self, inputs, targs):
        rnn_states, _ = self(inputs, targs)
        return rnn_states

    def instance_losses(self, inputs, targs):
        _, instance_losses = self(inputs, targs)
        return instance_losses

    def loss(self, inputs, targs):
        _, instance_losses = self(inputs, targs)
        total_loss = torch.mean(instance_losses)
        return total_loss
