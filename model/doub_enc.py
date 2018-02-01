import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import LSTMCell
from model.utils import load_embeddings, pad_sequence


class DoubEnc(object):
    def __init__(self, config):
        self.config = config

    def _add_placeholders(self):
        # shape = (batch_size, max length of sentence in batch)
        self.s1 = tf.placeholder(tf.int32, shape=[None, None], name='sent1')
        self.s2 = tf.placeholder(tf.int32, shape=[None, None], name='sent2')  # DoubEnc doesn't need this one
        # shape = (batch_size, 3, max length of sentence in batch)
        self.desc = tf.placeholder(tf.int32, shape=[None, None, None], name='descriptions')
        # shape = (batch_size, 1)
        self.idx = tf.placeholder(tf.int32, shape=[None], name='blank_index')
        # shape = (batch_size, 3)
        self.cand = tf.placeholder(tf.int32, shape=[None, None], name='candidates')
        # shape = (batch_size, 1)
        self.y = tf.placeholder(tf.int32, shape=[None, None], name='answer')

    def _get_feed_dict(self, s1, s2, idx, desc, cand, y):
        s1, seq_len1 = pad_sequence(s1, pad_tok=0, pad_left=True, nlevels=1)  # only pad words, not for chars
        s2, seq_len2 = pad_sequence(s2, pad_tok=0, pad_left=True, nlevels=1)
        desc, _ = pad_sequence(desc, pad_tok=0, pad_left=True, nlevels=2)
        pass  # TODO performs pad

    def _add_embedding_op(self):
        with tf.variable_scope('embeddings'):
            _word_embedding = tf.Variable(load_embeddings(self.config.embedding_path), name='_word_embeddings',
                                          dtype=tf.float32, trainable=self.config.finetune_emb)
            self.s1_emb = tf.nn.embedding_lookup(_word_embedding, self.s1, name='sent1_emb')
            self.s2_emb = tf.nn.embedding_lookup(_word_embedding, self.s2, name='sent2_emb')
            self.desc_emb = tf.nn.embedding_lookup(_word_embedding, self.desc, name='desc_emb')

    def _build_model_op(self):
        with tf.variable_scope('lexical_encoder'):
            lexical_cell = LSTMCell(num_units=self.config.num_units, state_is_tuple=True, use_peepholes=True)
            s = tf.shape(self.desc_emb)
            self.desc_emb = tf.reshape(self.desc_emb, shape=[s[0] * s[1], s[2], s[-1]])
            _, de = dynamic_rnn(lexical_cell, self.desc_emb, sequence_length=None, dtype=tf.float32, scope='lex_rnn')
            de = tf.reshape(de, shape=[s[0], s[1], self.config.num_units])  # (batch_size, num_cands, num_units)

        with tf.variable_scope('concatenate_sentence'):
            pass

        with tf.variable_scope('context_encoder'):
            context_cell = LSTMCell(num_units=self.config.num_units, state_is_tuple=True, use_peepholes=True)
            # TODO
        '''with tf.variable_scope('model'):
            tf.get_variable_scope().reuse_variables()
            lexical_cell = LSTMCell(num_units=self.config.num_units, state_is_tuple=True, use_peepholes=True)
            context_cell = LSTMCell(num_units=self.config.num_units, state_is_tuple=True, use_peepholes=True)

            # each with shape = (batch_size, max_length of sentence in batch, word_emb_dim)
            unstack_cand_sent_emb = tf.unstack(self.cand_sent_emb, axis=1)
            for emb in unstack_cand_sent_emb:
                _, d_e = dynamic_rnn(lexical_cell, emb, sequence_length=None, dtype=tf.float32,
                                     scope='lexical_encoder')
                tmp_state = tf.reshape(d_e, [-1, 1, self.config.num_units])
                s1_emb = tf.concat([self.s1l_emb, tmp_state, self.s1r_emb], axis=1)
                _, h_i = dynamic_rnn(context_cell, s1_emb, sequence_length=None, dtype=tf.float32,
                                     scope='context_encoder')'''


