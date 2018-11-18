import collections

from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import rnn_cell_impl
import tensorflow as tf

class SeqMatchSeqAttentionState(collections.namedtuple("SeqMatchSeqAttentionState", ("cell_state", "attention"))):
    pass

def get_hidden_state(cell_state):
    """ Get the hidden state needed in cell state which is
        possibly returned by LSTMCell, GRUCell, RNNCell or MultiRNNCell.

    Args:
      cell_state: a structure of cell state

    Returns:
      hidden_state: A Tensor
    """

    if type(cell_state) is tuple:
        cell_state = cell_state[-1]
    if hasattr(cell_state, "h"):
        hidden_state = cell_state.h
    else:
        hidden_state = cell_state
    return hidden_state

class SeqMatchSeqAttention(object):
    """ Attention for SeqMatchSeq.
  """

    def __init__(self, num_units, premise_mem, premise_mem_weights, name="SeqMatchSeqAttention"):
        """ Init SeqMatchSeqAttention

    Args:
      num_units: The depth of the attention mechanism.
      premise_mem: encoded premise memory
      premise_mem_weights: premise memory weights
    """
        # Init layers
        self._name = name
        self._num_units = num_units
        # Shape: [batch_size,max_premise_len,rnn_size]
        self._premise_mem = premise_mem
        # Shape: [batch_size,max_premise_len]
        self._premise_mem_weights = premise_mem_weights

        with tf.name_scope(self._name):
            self.query_layer = layers_core.Dense(num_units, name="query_layer", use_bias=False)
            self.hypothesis_mem_layer = layers_core.Dense(num_units, name="hypothesis_mem_layer", use_bias=False)
            self.premise_mem_layer = layers_core.Dense(num_units, name="premise_mem_layer", use_bias=False)
            # Preprocess premise Memory
            # Shape: [batch_size, max_premise_len, num_units]
            self._keys = self.premise_mem_layer(premise_mem)
            self.batch_size = self._keys.shape[0].value
            self.alignments_size = self._keys.shape[1].value

    def __call__(self, hypothesis_mem, query):
        """ Perform attention

    Args:
      hypothesis_mem: hypothesis memory
      query: hidden state from last time step

    Returns:
      attention: computed attention
    """
        with tf.name_scope(self._name):
            # Shape: [batch_size, 1, num_units]
            processed_hypothesis_mem = tf.expand_dims(self.hypothesis_mem_layer(hypothesis_mem), 1)
            # Shape: [batch_size, 1, num_units]
            processed_query = tf.expand_dims(self.query_layer(query), 1)
            v = tf.get_variable("attention_v", [self._num_units], dtype=tf.float32)
            # Shape: [batch_size, max_premise_len]
            score = tf.reduce_sum(v * tf.tanh(self._keys + processed_hypothesis_mem + processed_query), [2])
            # Mask score with -inf
            score_mask_values = float("-inf") * (1. - tf.cast(self._premise_mem_weights, tf.float32))
            masked_score = tf.where(tf.cast(self._premise_mem_weights, tf.bool), score, score_mask_values)
            # Calculate alignments
            # Shape: [batch_size, max_premise_len]
            alignments = tf.nn.softmax(masked_score)
            # Calculate attention
            # Shape: [batch_size, rnn_size]
            attention = tf.reduce_sum(tf.expand_dims(alignments, 2) * self._premise_mem, axis=1)
            return attention


# noinspection PyProtectedMember
class SeqMatchSeqWrapper(rnn_cell_impl.RNNCell):
    """ RNN Wrapper for SeqMatchSeq.
  """

    def __init__(self, cell, attention_mechanism, name='SeqMatchSeqWrapper'):
        super(SeqMatchSeqWrapper, self).__init__(name=name)
        self._cell = cell
        self._attention_mechanism = attention_mechanism

    # noinspection PyMethodOverriding
    def call(self, inputs, state, scope=None):
        """
    Args:
      inputs: inputs at some time step
      state: A (structure of) cell state
    """
        # Concatenate attention and input
        cell_inputs = tf.concat([state.attention, inputs], axis=-1)
        cell_state = state.cell_state
        # Call cell function
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        # Get hidden state
        hidden_state = get_hidden_state(cell_state)
        # Calculate attention
        attention = self._attention_mechanism(inputs, hidden_state)
        # Assemble next state
        next_state = SeqMatchSeqAttentionState(
            cell_state=next_cell_state,
            attention=attention)
        return cell_output, next_state

    @property
    def state_size(self):
        return SeqMatchSeqAttentionState(
            cell_state=self._cell.state_size,
            attention=self._attention_mechanism._premise_mem.get_shape()[-1].value
        )

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        cell_state = self._cell.zero_state(batch_size, dtype)
        attention = rnn_cell_impl._zero_state_tensors(self.state_size.attention, batch_size, tf.float32)
        return SeqMatchSeqAttentionState(
            cell_state=cell_state,
            attention=attention)