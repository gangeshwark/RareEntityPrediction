import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import LSTMCell
from model.utils import load_embeddings, pad_sequence, batch_iter
from data.data_prepro import max_sent_len, max_sent_len_desc


class DoubEnc(object):
    def __init__(self, config):
        self.config = config
        self._add_placeholders()
        self._add_embedding_op()
        self._build_model_op()
        self._build_loss_op()
        self._build_pred_op()
        self._build_train_op()
        self.sess = None
        self.saver = None
        self.initialize_session()

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

    def _add_placeholders(self):
        # shape = (batch_size, max length of sentence in batch)
        self.s1 = tf.placeholder(tf.int32, shape=[None, None], name='sent1')
        # shape = (batch_size)
        self.s1_seq_len = tf.placeholder(tf.int32, shape=[None], name='sent1_seq_length')
        # shape = (batch_size, max length of sentence in batch)
        # self.s2 = tf.placeholder(tf.int32, shape=[None, None], name='sent2')  # DoubEnc doesn't need this one
        # shape = (batch_size)
        # self.s2_seq_len = tf.placeholder(tf.int32, shape=[None], name='sent2_seq_length')
        # shape = (batch_size, num_cands, max length of sentence in batch)
        self.desc = tf.placeholder(tf.int32, shape=[None, None, None], name='descriptions')
        # shape = (batch_size, num_cands)
        self.desc_seq_len = tf.placeholder(tf.int32, shape=[None, None], name='desc_seq_length')
        # shape = (batch_size)
        self.idx = tf.placeholder(tf.int32, shape=[None], name='blank_index')
        # shape = (batch_size, num_candidates)
        self.cand = tf.placeholder(tf.int32, shape=[None, None], name='candidates')
        # shape = (batch_size, num_candidates)
        self.y = tf.placeholder(tf.int32, shape=[None, None], name='answer')

    def _get_feed_dict(self, s1, s2, desc, idx, cand, y=None):
        s1, s1_seq_len = pad_sequence(s1, max_length=max_sent_len - 1, pad_tok=0, pad_left=True, nlevels=1)
        s1_seq_len = [x + 1 for x in s1_seq_len]
        s2, s2_seq_len = pad_sequence(s2, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        desc, desc_seq_len = pad_sequence(desc, max_length=max_sent_len_desc, pad_tok=0, pad_left=True, nlevels=2)
        feed_dict = {
            self.s1: s1, self.s1_seq_len: s1_seq_len,
            self.desc: desc, self.desc_seq_len: desc_seq_len,
            self.idx: idx,
            self.cand: cand}
        if y is not None:
            feed_dict[self.y] = y
        return feed_dict

    def _add_embedding_op(self):
        with tf.variable_scope('embeddings'):
            _word_embedding = tf.Variable(load_embeddings(self.config.embedding_path), name='_word_embeddings',
                                          dtype=tf.float32, trainable=self.config.finetune_emb)
            # self.s1l_emb = tf.nn.embedding_lookup(_word_embedding, self.s1l, name='sent1_left_emb')
            # self.s1r_emb = tf.nn.embedding_lookup(_word_embedding, self.s1r, name='sent1_right_emb')
            self.s1_emb = tf.nn.embedding_lookup(_word_embedding, self.s1, name='sent1_emb')
            # self.s2_emb = tf.nn.embedding_lookup(_word_embedding, self.s2, name='sent2_emb')
            self.desc_emb = tf.nn.embedding_lookup(_word_embedding, self.desc, name='desc_emb')

    def _build_model_op(self):
        with tf.variable_scope('lexical_encoder'):
            lexical_cell = LSTMCell(num_units=self.config.num_units, state_is_tuple=True, use_peepholes=True)
            s = tf.shape(self.desc_emb)
            desc_emb = tf.reshape(self.desc_emb, shape=[s[0] * s[1], s[2], s[-1]])
            desc_seq_len = tf.reshape(self.desc_seq_len, shape=[s[0] * s[1]])
            _, de = dynamic_rnn(lexical_cell, desc_emb, sequence_length=desc_seq_len, dtype=tf.float32, scope='lex_rnn')
            self.de = tf.reshape(de, shape=[s[0], s[1], self.config.num_units])  # (batch_size, num_cands, num_units)

        '''with tf.variable_scope('concatenate_sentence'):
            de = tf.expand_dims(self.de, axis=2)  # (batch_size, num_cands, 1, num_units)
            de_shape = tf.shape(de)
            # (batch_size, num_cands, sentence_length, word_dim)
            s1l_emb = tf.tile(tf.expand_dims(self.s1l_emb, axis=1), [1, de_shape[1], 1, 1])
            s1r_emb = tf.tile(tf.expand_dims(self.s1r_emb, axis=1), [1, de_shape[1], 1, 1])
            s1_emb = tf.concat([s1l_emb, de, s1r_emb], axis=-2)'''

        with tf.variable_scope('concat_sentence'):
            de_list = tf.unstack(self.de, axis=1)  # each: (batch_size, num_units)
            merged_s1_list = []
            for de_res in de_list:
                s1_emb_list = tf.unstack(self.s1_emb, axis=0)  # each: (sent_length, dim)
                idx_list = tf.unstack(self.idx, axis=0)  # each: (1)
                de_res_list = tf.unstack(de_res, axis=0)  # each: (num_units)
                tmp_merged = []
                for i in range(len(idx_list)):
                    lsz = max_sent_len - 1 - idx_list[i]  # left sent size
                    rsz = idx_list[i]  # right sent size
                    s1_emb_left, s1_emb_right = tf.split(s1_emb_list[i], num_or_size_splits=[lsz, rsz], axis=0)
                    de_tmp = tf.expand_dims(de_res_list[i], axis=0) # (1, num_units)
                    s1 = tf.concat([s1_emb_left, de_tmp, s1_emb_right], axis=0)  # (max_sent_len, dim)
                    tmp_merged.append(s1)
                merged = tf.stack(tmp_merged)  # (batch_size, max_sent_len, dim)
                merged = tf.expand_dims(merged, axis=1)  # (batch_size, 1, max_sent_len, dim)
                merged_s1_list.append(merged)
            s1_emb = tf.concat(merged_s1_list, axis=1)  # (batch, num_cands, max_sent_len, dim)

        with tf.variable_scope('context_encoder'):
            context_cell = LSTMCell(num_units=self.config.num_units, state_is_tuple=True, use_peepholes=True)
            s = tf.shape(s1_emb)
            s1_emb = tf.reshape(s1_emb, shape=[s[0] * s[1], s[2], s[-1]])
            s1_seq_len = tf.concat([self.s1_seq_len for _ in range(s[1])], axis=0)
            _, hi = dynamic_rnn(context_cell, s1_emb, sequence_length=s1_seq_len, dtype=tf.float32, scope='con_rnn')
            self.hi = tf.reshape(hi, shape=[s[0], s[1], self.config.num_units])

        with tf.variable_scope('project'):
            batch_size, num_cands, _ = tf.shape(self.de)
            w = tf.get_variable(name='W', shape=[self.config.num_units, self.config.num_units], dtype=tf.float32)
            de = tf.transpose(tf.reshape(self.de, shape=[-1, self.config.num_units]))  # (num_units, batch_size * cands)
            hi = tf.reshape(self.hi, shape=[-1, self.config.num_units])  # (batch_size * cands, num_units)
            output = tf.matmul(tf.matmul(hi, w), de)  # ignore bias
            self.logits = tf.reshape(tf.reduce_sum(output, axis=1), shape=[batch_size, num_cands])

    def _build_loss_op(self):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, tf.float32))
        self.loss = tf.reduce_mean(losses)

    def _build_pred_op(self):
        self.preds = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def _build_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        if self.config.grad_clip is not None:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grands, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))

    def acc(self):
        correct_preds = tf.equal(tf.argmax(self.logits, axis=-1), tf.argmax(self.y, axis=-1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32))

    def train(self, dataset, devset, batch_size, epochs):
        for epoch in range(1, epochs + 1):
            for i, (s1, s2, idx, desc, cand, y) in enumerate(batch_iter(dataset, batch_size)):
                feed_dict = self._get_feed_dict(s1, s2, desc, idx, cand, y)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            # build evaluate
            self.evaluate(devset, batch_size)
        # TODO

    def evaluate(self, dataset, batch_size):
        # TODO
        nbatches = (len(dataset) + batch_size - 1) // batch_size
        acc = []
        for s1, s2, idx, desc, cand, y in batch_iter(dataset, batch_size):
            feed_dict = self._get_feed_dict(s1, s2, desc, idx, cand, y)
            batch_acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
            acc.append(batch_acc)
        print('Accuracy: {:04.2f}'.format(sum(acc) / nbatches * 100))

