import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as _bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn import dynamic_rnn as _dynamic_rnn


class bidirectional_dynamic_rnn:
    def __init__(self, num_units, use_peepholes=False, scope='bi_dynamic_rnn'):
        self.num_units = num_units
        self.cell_fw = LSTMCell(self.num_units, use_peepholes=use_peepholes)
        self.cell_bw = LSTMCell(self.num_units, use_peepholes=use_peepholes)
        self.scope = scope

    def __call__(self, inputs, seq_len, return_last_state=False, keep_prob=None, is_train=None):
        with tf.variable_scope(self.scope):
            if return_last_state:
                _, ((_, output_fw), (_, output_bw)) = _bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs,
                                                                                 sequence_length=seq_len,
                                                                                 dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
            else:
                (output_fw, output_bw), _ = _bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs,
                                                                       sequence_length=seq_len, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
            output = dropout(output, keep_prob, is_train)
            return output


class dynamic_rnn:
    def __init__(self, num_units, use_peepholes=False, scope='dynamic_rnn'):
        self.num_units = num_units
        self.cell = LSTMCell(self.num_units, use_peepholes=use_peepholes)
        self.scope = scope

    def __call__(self, inputs, seq_len, return_last_state=False, keep_prob=None, is_train=None):
        with tf.variable_scope(self.scope):
            if return_last_state:
                _, (_, output) = _dynamic_rnn(self.cell, inputs, sequence_length=seq_len, dtype=tf.float32)
            else:
                output, _ = _dynamic_rnn(self.cell, inputs, sequence_length=seq_len, dtype=tf.float32)
            output = dropout(output, keep_prob, is_train)
            # print("Output", output.get_shape())
            return output


def dense(inputs, hidden_dim, use_bias=True, scope='dense'):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden_dim]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        w = tf.get_variable("W", shape=[dim, hidden_dim], dtype=tf.float32)
        output = tf.matmul(flat_inputs, w)
        if use_bias:
            b = tf.get_variable("b", shape=[hidden_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(output, b)
        output = tf.reshape(output, out_shape)
        return output


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if keep_prob is not None and is_train is not None:
            out = tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed), lambda: x)
            return out
        return x
