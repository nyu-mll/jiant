import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class LeftRightDecoder(TestDecoder):
    @classmethod
    def _initialize(cls, decoder_cell, examples):
        """Initialize RNN and decoder states.

        Args:
            decoder_cell (DecoderCell)
            examples (list[Example])

        Returns:
            rnn_state (RNNState)
            states (list[DecoderState])
        """
        states = [DecoderState.initial(ex) for ex in examples]
        batch_size = len(examples)
        rnn_state = decoder_cell.initialize(batch_size)
        # TODO: make hidden states volatile

        return rnn_state, states

    @classmethod
    def _advance_rnn(self, token_embedder, decoder_cell, rnn_context_combiner, encoder_output, rnn_state, states):
        """Advance the RNN state.

        Args:
            token_embedder (TokenEmbedder)
            decoder_cell (DecoderCell)
            rnn_context_combiner (RNNContextCombiner)
            encoder_output (EncoderOutput)
            rnn_state (RNNState)
            states (list[DecoderState])

        Returns:
            rnn_state (RNNState)
            predictions (PredictionBatch)
        """

        # apply decoder cell, update hidden states
        previous_words = [state.token for state in states]  # get latest words
        advance = GPUVariable(torch.ones(len(states), 1))

        # advance the RNN
        x = token_embedder.embed_tokens(previous_words)
        rnn_input = rnn_context_combiner(encoder_output, x)
        dc_output = decoder_cell(rnn_state, rnn_input, advance)
        rnn_state = dc_output.rnn_state

        # make predictions for the entire batch
        predictions = dc_output.predictions
        vocab = dc_output.vocab
        token_probs = predictions.probs  # (batch_size, vocab_size)

        # terminated sequence can only be extended with <stop>
        terminated_indices = np.array([i for i, state in enumerate(states) if state.terminated], dtype=np.int32)
        stop_idx = vocab.word2index(WordVocab.STOP)
        token_probs[terminated_indices, :] = 0.0
        token_probs[terminated_indices, stop_idx] = 1.0
        # NOTE: we have modified token_probs in-place. Hence, we have modified predictions in-place.

        return rnn_state, predictions

    @classmethod
    def _prefix_hints_at_time_t(cls, prefix_hints, t, beam_size):
        """Extract prefix hints at time t.

        Args:
            prefix_hints (list[list[unicode]]): a batch of prefixes, one per example.
            t (int): the time index to pull out
            beam_size (int)

        Returns:
            list[unicode]: prefix hints at time t, of length len(prefix_hints) * beam_size.
        """
        hints_at_t = []
        for prefix in prefix_hints:
            if t >= len(prefix):
                hint = None
            else:
                hint = prefix[t]
            hints_at_t.extend([hint] * beam_size)
        return hints_at_t

    @classmethod
    def _recover_sequences(cls, states_over_time, beam_size, top_k):
        # create decoder_traces
        ex_idx_to_beam_traces = defaultdict(list)
        for t, states in enumerate(states_over_time):
            assert len(states) % beam_size == 0
            beams = list(chunks(states, beam_size))
            for ex_idx, beam in enumerate(beams):
                trace = BeamTrace(beam, top_k)
                ex_idx_to_beam_traces[ex_idx].append(trace)

        decoder_traces = []
        for ex_idx in range(max(ex_idx_to_beam_traces.keys()) + 1):
            beam_traces = ex_idx_to_beam_traces[ex_idx]
            decoder_traces.append(BeamDecoderTrace(beam_traces))

        final_state_beams = list(chunks(states_over_time[-1], beam_size))
        output_beams = [[state.token_sequence for state in state_beam] for state_beam in final_state_beams]

        return output_beams, decoder_traces




class SampleDecoder(LeftRightDecoder):
    def __init__(self, decoder_cell, token_embedder, rnn_context_combiner):
        """Construct BeamDecoder.

        Args:
            decoder_cell (DecoderCell)
            token_embedder (TokenEmbedder)
            rnn_context_combiner (RNNContextCombiner)

        """
        self.decoder_cell = decoder_cell
        self.word_vocab = token_embedder.vocab
        self.token_embedder = token_embedder
        self.word_dim = token_embedder.embed_dim
        self.rnn_context_combiner = rnn_context_combiner

    def decode(self, examples, encoder_output, beam_size, prefix_hints, max_seq_length=50, top_k=5, temperature=1.):
        """Sample an output. 

        Args:
            examples (list[Example])
            encoder_output (EncoderOutput)
            beam_size (int)
            prefix_hints (list[list[unicode]]): a batch of prefixes. For each example, all returned results will start
                with the specified prefix.
            max_seq_length (int): maximum allowable length of outputted sequences
            top_k (int): number of beam candidates to show in trace
            temperature (float): sampling temperature

        Returns:
            beams (list[list[list[unicode]]]): a batch of beams of decoded sequences
            traces (list[PredictionTrace])
        """
        rnn_state_orig, states_orig = self._initialize(self.decoder_cell, examples)

        # duplicate everything to beam_size
        duplicate = BeamDuplicator(beam_size)
        rnn_state = duplicate(rnn_state_orig)
        encoder_output = duplicate(encoder_output)
        states = []
        for state in states_orig:
            states.extend([state] * beam_size)

        states_over_time = []
        for t in range(max_seq_length):
            # stop if all sequences have terminated
            if all(state.terminated for state in states): break
            hints_at_t = self._prefix_hints_at_time_t(prefix_hints, t, beam_size)
            rnn_state, states = self._advance(encoder_output, rnn_state, states, hints_at_t, temperature)
            states_over_time.append(states)

        return self._recover_sequences(states_over_time, beam_size, top_k=top_k)

    def _advance(self, encoder_output, rnn_state, states, hints_at_t, temperature):
        """

        Args:
            encoder_output (EncoderOutput)
            rnn_state (RNNState)
            states (list[DecoderState]) 
            hints_at_t (list[unicode]): of length = len(states)
            temperature (float)

        Returns:
            rnn_state (RNNState)
            states (list[DecoderState])
        """
        # update RNN state
        rnn_state, predictions = self._advance_rnn(self.token_embedder, self.decoder_cell, self.rnn_context_combiner,
                                                   encoder_output, rnn_state, states)
        token_probs, vocab = predictions
        vocab_size = len(vocab)

        # update states
        new_states = []
        for batch_idx, (state, hint) in enumerate(izip(states, hints_at_t)):
            if state.terminated:
                new_state = state
            else:
                if hint is None:
                    sampling_probs = token_probs[batch_idx]  # np.array (vocab_size,)
                    sampling_probs = temperature_smooth(sampling_probs, temperature)
                    token_idx = np.random.choice(vocab_size, p=sampling_probs)  # select token according to prob
                else:
                    token_idx = vocab.word2index(hint)  # follow the hint
                token = vocab.index2word(token_idx)
                token_prob = token_probs[batch_idx, token_idx]
                extension_prob = state.sequence_prob * token_prob  # compute prob of entire new sequence

                candidates = [Candidate(token, token_prob)]
                trace = PredictionTrace(candidates, [])
                new_state = state.extend(token, extension_prob, trace)

            new_states.append(new_state)

        return rnn_state, new_states


class BeamDecoder(LeftRightDecoder):
    def __init__(self, decoder_cell, token_embedder, rnn_context_combiner):
        """Construct BeamDecoder.

        Args:
            decoder_cell (DecoderCell)
            token_embedder (TokenEmbedder)
            rnn_context_combiner (RNNContextCombiner)

        """
        self.decoder_cell = decoder_cell
        self.word_vocab = token_embedder.vocab
        self.token_embedder = token_embedder
        self.word_dim = token_embedder.embed_dim
        self.rnn_context_combiner = rnn_context_combiner

    def decode(self, examples, encoder_output, weighted_value_estimators,
               beam_size, prefix_hints, sibling_penalty, max_seq_length=50, top_k=5, verbose=False):
        """Beam decode.

        Args:
            examples (list[Example])
            encoder_output (EncoderOutput)
            weighted_value_estimators (list[(ValueEstimator, float)]): a list of (estimator, weight) pairs.
            beam_size (int)
            prefix_hints (list[list[unicode]]): a batch of prefixes. For each example, all returned results will start
                with the specified prefix.
            sibling_penalty (float)
            max_seq_length (int): maximum allowable length of outputted sequences
            top_k (int): number of beam candidates to show in trace
            verbose (bool): default is False

        Returns:
            beams (list[list[list[unicode]]]): a batch of beams of decoded sequences
            traces (list[BeamDecoderTrace])
        """
        rnn_state_orig, states_orig = self._initialize(self.decoder_cell, examples)

        # duplicate everything to beam_size
        duplicate = BeamDuplicator(beam_size)
        rnn_state = duplicate(rnn_state_orig)
        encoder_output = duplicate(encoder_output)

        states = []
        for state in states_orig:
            states.append(state)
            # these states are guaranteed to die on the first round, because their sequence_prob = 0
            # they are just here as padding
            # TODO(kelvin): WARNING! In the future, the ValueEstimators in BeamDecoder._advance might break
            # my assumption that any extension of a sequence with 0 prob will also have 0 prob.
            # If this assumption is broken, the BeamDecoder will return a beam of identical results.
            doomed = [DecoderState.initial_doomed(state.example)] * (beam_size - 1)
            states.extend(doomed)

        # perform iterations of beam search
        time_steps = range(max_seq_length)
        if verbose:
            time_steps = verboserate(time_steps, desc='Beam decoding sequences')

        states_over_time = []
        for _ in time_steps:
            # stop if all sequences have terminated
            if all(state.terminated for state in states): break
            rnn_state, states = self._advance(encoder_output, weighted_value_estimators, beam_size, rnn_state, states,
                                              sibling_penalty)
            states_over_time.append(states)

        return self._recover_sequences(states_over_time, beam_size, top_k)

    @classmethod
    def _select_extensions_fast(cls, extension_probs, beam_size):
        extension_probs_sorted, original_indices = cls._truncate_extension_probs(extension_probs,
                                                                                 beam_size)  # (batch_size, beam_size)
        batch_indices, sorted_token_indices = cls._select_extensions(extension_probs_sorted,
                                                                     beam_size)  # 1D array of batch_size * beam_size
        token_indices = original_indices[batch_indices, sorted_token_indices]  # 1D array of batch_size * beam_size
        return batch_indices, token_indices

    @classmethod
    def _select_extensions(cls, extension_probs, beam_size):
        """For each beam in extension_probs, select <beam_size> elements to continue.

        Args:
            extension_probs (np.ndarray): of shape (batch_size, vocab_size). Containing the probability of
                every extension of every element in the batch.
            beam_size (int): must satisfy batch_size % beam_size == 0

        Returns:
            batch_indices (np.ndarray): 1D array, batch indices of the top extensions
            token_indices (np.ndarray): 1D array, token indices of the top extensions
        """
        batch_size, vocab_size = extension_probs.shape
        num_beams = batch_size / beam_size
        assert batch_size % beam_size == 0

        beam_probs = np.reshape(extension_probs, (num_beams, vocab_size * beam_size))
        top_indices = np.argsort(-beam_probs, axis=1)  # TODO: do this in PyTorch
        top_indices = top_indices[:, :beam_size]  # (num_beams, beam_size)

        assert top_indices.dtype == np.int64  # going to do int arithmetic with this

        beam_indices = np.expand_dims(np.arange(num_beams), 1)  # (num_beams, 1)

        batch_indices = beam_indices * beam_size + top_indices / vocab_size
        token_indices = top_indices % vocab_size

        return batch_indices.flatten(), token_indices.flatten()

    @classmethod
    def _truncate_extension_probs(cls, extension_probs, beam_size):
        """For each example, keep only the k highest scoring extension probs.

        Where k = beam_size.

        Args:
            extension_probs (np.ndarray): of shape (batch_size, vocab_size)
            beam_size (int)

        Returns:
            extension_probs_sorted (np.ndarray): of shape (batch_size, beam_size). Like extension_probs, but each
                row is sorted in descending probability, and truncated to a length of beam_size.
            original_indices (np.ndarray): of shape (batch_size, beam_size).
                original_indices[i, j] = the original column index of the probability value at extension_probs_sorted[i, j]
        """
        extension_probs_var = try_gpu(Variable((torch.from_numpy(extension_probs)), volatile=True))
        extension_probs_sorted_var, original_indices_var = torch.sort(extension_probs_var, 1, descending=True)
        extension_probs_sorted_var = extension_probs_sorted_var[:, :beam_size]
        original_indices_var = original_indices_var[:, :beam_size]

        from_var = lambda v: v.data.cpu().numpy()
        extension_probs_sorted = from_var(extension_probs_sorted_var)
        original_indices = from_var(original_indices_var)

        # batch_size, vocab_size = extension_probs.shape
        # original_indices = np.argsort(-extension_probs, axis=1)  # (batch_size, vocab_size)
        # original_indices = original_indices[:, :beam_size]  # (batch_size, beam_size)
        #
        # j_indices, i_indices = np.meshgrid(np.arange(beam_size), np.arange(batch_size))  # (batch_size, beam_size)
        #
        # extension_probs_sorted = extension_probs[i_indices, original_indices]  # (batch_size, beam_size)

        return extension_probs_sorted, original_indices

    @classmethod
    def penalize_extensions_by_rank(cls, extension_probs, penalty):
        """Penalize extensions by their rank, as done in Li et al. 2016.

        "A Simple, Fast Diverse Decoding Algorithm for Neural Generation."

        Args:
            extension_probs (np.ndarray): of shape (batch_size, vocab_size)
            penalty (float)
        """
        batch_size, vocab_size = extension_probs.shape
        penalized_extension_probs = np.copy(extension_probs)

        if penalty == 0.0:
            return penalized_extension_probs  # shortcut for when there is no penalty

        top_indices = np.argsort(-extension_probs, axis=1)
        j_indices, i_indices = np.meshgrid(np.arange(vocab_size), np.arange(batch_size))
        penalized_extension_probs[i_indices, top_indices] /= np.exp(penalty * j_indices)
        return penalized_extension_probs

    def _advance(self, encoder_output, weighted_value_estimators, beam_size, rnn_state, states, sibling_penalty):
        """Take one step of beam search.

        Args:
            encoder_output (EncoderOutput)
            weighted_value_estimators (list[(ValueEstimator, float)]): a list of (estimator, weight) pairs.
            beam_size (int)
            rnn_state (RNNState)
            states (list[DecoderState])
            sibling_penalty (float)

        Returns:
            h (Variable): (batch_size, hidden_dim)
            c (Variable): (batch_size, hidden_dim)
            states (list[DecoderState])
        """
        rnn_state, predictions = self._advance_rnn(self.token_embedder, self.decoder_cell, self.rnn_context_combiner,
                                                   encoder_output, rnn_state, states)
        token_probs, vocab = predictions

        sequence_probs = np.expand_dims(np.array([s.sequence_prob for s in states]), 1)  # (batch_size, 1)
        extension_probs = sequence_probs * token_probs  # (batch_size, vocab_size)

        # modify extension probs using value estimators
        modified_extension_probs = np.copy(extension_probs)
        for val_estimator, weight in weighted_value_estimators:
            modified_extension_probs *= np.exp(weight * val_estimator.value(states, rnn_state))

        # apply diversity-inducing sibling penalization trick
        modified_extension_probs = self.penalize_extensions_by_rank(modified_extension_probs, sibling_penalty)

        # select the best extensions of each beam, using MODIFIED extension_probs
        batch_indices, token_indices = self._select_extensions_fast(modified_extension_probs, beam_size)
        # both batch_indices and token_indices are (batch_size,)

        # select surviving RNN states
        batch_selector = BatchSelector(batch_indices)
        rnn_state = batch_selector(rnn_state)

        # update states
        # note that here we store the original, UN-MODIFIED extension_probs
        # these are actual generation probabilities
        new_states = []
        for batch_idx, token_idx in izip(batch_indices, token_indices):
            state = states[batch_idx]
            if state.terminated:
                new_state = state
            else:
                token = vocab.index2word(token_idx)
                extension_prob = extension_probs[batch_idx, token_idx]

                # construct trace
                token_prob = token_probs[batch_idx, token_idx]
                candidates = [Candidate(token, token_prob)]
                trace = PredictionTrace(candidates, [])
                # TODO(kelvin): add more info to trace

                new_state = state.extend(token, extension_prob, trace)

            new_states.append(new_state)

        return rnn_state, new_states

