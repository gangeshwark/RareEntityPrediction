import tensorflow as tf
from dataset.data_prepro import max_sent_len
from model.utils import pad_sequence, batch_iter
from model.nns import bidirectional_dynamic_rnn
from model.logger import Progbar


class Model(object):
    def __init__(self, config):
        self.config = config
        self._add_placeholders()
        self._build_model_op()
        self._build_loss_op()
        self._build_train_op()
        self._compute_accuracy()
        self.sess, self.saver = None, None
        self.initialize_session()

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

    def save_session(self, epoch):
        self.saver.save(self.sess, self.config.ckpt_path + self.config.model_name, global_step=epoch)

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.config.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def close_session(self):
        self.sess.close()

    def _add_placeholders(self):
        # shape = (batch_size, max length of sentence in batch)
        self.sl = tf.placeholder(tf.int32, shape=[self.config.batch_size, max_sent_len], name='sl')
        self.sr = tf.placeholder(tf.int32, shape=[self.config.batch_size, max_sent_len], name='sr')
        # shape = (batch_size)
        self.sent_seq_len = tf.placeholder(tf.int32, shape=[self.config.batch_size], name='sent_seq_length')
        # shape = (batch_size, num_cands, max length of sentence in batch)
        self.desc = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.num_cands, max_sent_len],
                                   name='desc')
        # shape = (batch_size, num_cands)
        self.desc_seq_len = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.num_cands],
                                           name='desc_seq_length')
        # shape = (batch_size, num_candidates)
        self.cand = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.num_cands], name='candidates')
        # shape = (batch_size, num_candidates)
        self.y = tf.placeholder(tf.int32, shape=[self.config.batch_size, self.config.num_cands], name='answer')
        # hyperparameters
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.lr = tf.placeholder(dtype=tf.float32, name='lr')
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')

    def _get_feed_dict(self, sl, sr, desc, cand, is_train, y=None, lr=None, keep_prob=None):
        sl, sl_seq_len = pad_sequence(sl, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        sr, sr_seq_len = pad_sequence(sr, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        sent_seq_len = [x + y + 1 for x, y in zip(sl_seq_len, sr_seq_len)]
        desc, desc_seq_len = pad_sequence(desc, max_length_2=max_sent_len, pad_tok=0, pad_left=True, nlevels=2)
        feed_dict = {self.sl: sl, self.sr: sr, self.sent_seq_len: sent_seq_len,
                     self.desc: desc, self.desc_seq_len: desc_seq_len,
                     self.cand: cand, self.is_train: is_train}
        if y is not None:
            feed_dict[self.y] = y
        if lr is not None:
            feed_dict[self.lr] = lr
        if keep_prob is not None:
            feed_dict[self.keep_prob] = keep_prob
        return feed_dict

    def _build_model_op(self):
        with tf.variable_scope('embeddings'):
            _word_emb = tf.Variable(self.config.emb, name='_word_emb', dtype=tf.float32,
                                    trainable=self.config.finetune_emb)
            self.sl_emb = tf.nn.embedding_lookup(_word_emb, self.sl, name='sent_left_emb')
            self.sr_emb = tf.nn.embedding_lookup(_word_emb, self.sr, name='sent_right_emb')
            self.desc_emb = tf.nn.embedding_lookup(_word_emb, self.desc, name='desc_emb')

        with tf.variable_scope('lexical_encoder'):
            s = self.desc_emb.get_shape()
            desc_emb = tf.reshape(self.desc_emb, shape=[s[0] * s[1], s[2], s[-1]])
            desc_seq_len = tf.reshape(self.desc_seq_len, shape=[s[0] * s[1]])
            lexical_rnn = bidirectional_dynamic_rnn(self.config.desc_units, use_peepholes=True, scope='lexical_rnn')
            de = lexical_rnn(desc_emb, desc_seq_len, return_last_state=True)
            # (batch_size, num_cands, num_units)
            self.de = tf.reshape(de, shape=[s[0], s[1], 2 * self.config.desc_units])

        with tf.variable_scope('concat_sentence'):
            de = tf.expand_dims(self.de, axis=2)  # (batch_size, num_cands, 1, num_units)
            print("de", de.get_shape())
            sl_emb = tf.tile(tf.expand_dims(self.sl_emb, axis=1), [1, self.config.num_cands, 1, 1])
            sr_emb = tf.tile(tf.expand_dims(self.sl_emb, axis=1), [1, self.config.num_cands, 1, 1])
            s_emb = tf.concat([sl_emb, de, sr_emb], axis=-2)
            print("s_emb", s_emb.get_shape())

        with tf.variable_scope('context_encoder'):
            s = s_emb.get_shape().as_list()
            s_emb = tf.reshape(s_emb, shape=[s[0] * s[1], s[2], s[3]])
            sent_seq_len = tf.concat([self.sent_seq_len for _ in range(s[1])], axis=0)
            context_rnn = bidirectional_dynamic_rnn(self.config.num_units, use_peepholes=True, scope='context_rnn')
            hi = context_rnn(s_emb, sent_seq_len, return_last_state=True)
            self.hi = tf.reshape(hi, shape=[s[0], s[1], 2 * self.config.num_units])

        with tf.variable_scope('project'):
            batch_size, num_cands, _ = self.de.get_shape()
            w = tf.get_variable(name='W', shape=[batch_size, 2 * self.config.num_units, 2 * self.config.num_units],
                                dtype=tf.float32)
            # (2 * num_units, batch_size * cands)
            # de = tf.transpose(tf.reshape(self.de, shape=[-1, 2 * self.config.num_units * s[0]]))
            # hi = tf.reshape(self.hi, shape=[-1, 2 * self.config.num_units * 3])  # (batch_size * cands, 2 * num_units)
            # p = tf.diag_part(tf.matmul(tf.matmul(hi, w), de))
            de = tf.transpose(self.de, perm=[0, 2, 1])
            print(de.get_shape())
            p = tf.matmul(tf.matmul(self.hi, w), de)
            print("p", p.get_shape())
            output = tf.sigmoid(p)  # ignore bias
            print("output", output.get_shape())
            print("y", self.y.get_shape())
            self.logits = tf.reshape(p, shape=[batch_size, num_cands])

    def _build_loss_op(self):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, tf.float32))
        self.loss = tf.reduce_mean(losses)

    def _build_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.config.grad_clip is not None:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grands, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))

    def _compute_accuracy(self):
        correct_preds = tf.equal(tf.argmax(self.logits, axis=-1), tf.argmax(self.y, axis=-1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32))

    def train(self, dataset, devset, sub_set, epochs):
        self.config.logger.info('Start training...')
        nbatches = (len(dataset) + self.config.batch_size - 1) // self.config.batch_size
        for epoch in range(1, epochs + 1):
            self.config.logger.info('Epoch %2d/%2d:' % (epoch, epochs))
            prog = Progbar(target=nbatches)  # nbatches
            for i, (sl, sr, desc, cand, y) in enumerate(batch_iter(dataset, self.config.batch_size)):
                feed_dict = self._get_feed_dict(sl, sr, desc, cand, True, y=y, lr=self.config.lr,
                                                keep_prob=self.config.keep_prob)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                prog.update(i + 1, [("train loss", train_loss)])
                if i % 1000 == 0:
                    self.evaluate(sub_set, batch_size=self.config.batch_size)
            self.config.lr *= self.config.lr_decay
            # build evaluate
            self.evaluate(devset, self.config.batch_size)

    def evaluate(self, dataset, batch_size):
        nbatches = (len(dataset) + batch_size - 1) // batch_size
        acc = []
        for sl, sr, desc, cand, y in batch_iter(dataset, batch_size):
            feed_dict = self._get_feed_dict(sl, sr, desc, cand, False, y=y, lr=self.config.lr, keep_prob=1.0)
            batch_acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
            acc.append(batch_acc)
        # assert len(acc) == nbatches
        self.config.logger.info('\nAccuracy: {:04.2f}'.format((sum(acc) / len(acc)) * 100))
