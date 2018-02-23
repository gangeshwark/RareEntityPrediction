import tensorflow as tf
from dataset.data_prepro import max_sent_len
from model.utils_new import pad_sequence, batch_iter
from model.nns import bidirectional_dynamic_rnn, dense
from model.logger import Progbar
import numpy as np


class Model(object):
    def __init__(self, config):
        self.config = config
        self._add_placeholders()
        self._build_model_op()
        self._build_loss_op()
        self._build_train_op()
        '''self._compute_accuracy()'''
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
        self.sl = tf.placeholder(tf.int32, shape=[None, max_sent_len], name='sl')
        self.sr = tf.placeholder(tf.int32, shape=[None, max_sent_len], name='sr')
        # shape = (batch_size)
        self.sent_seq_len = tf.placeholder(tf.int32, shape=[None], name='sent_seq_length')
        # shape = (batch_size, max length of sentence in batch)
        self.desc = tf.placeholder(tf.int32, shape=[None, max_sent_len], name='desc')
        # shape = (batch_size)
        self.desc_seq_len = tf.placeholder(tf.int32, shape=[None], name='desc_seq_length')
        # shape = (batch_size)
        self.cand = tf.placeholder(tf.int32, shape=[None], name='candidates')
        # shape = (batch_size)
        self.y = tf.placeholder(tf.int32, shape=[None], name='answer')
        # hyperparameters
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.lr = tf.placeholder(dtype=tf.float32, name='lr')
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')

    def _get_feed_dict(self, sl, sr, desc, cand, is_train, y=None, lr=None, keep_prob=None):
        sl, sl_seq_len = pad_sequence(sl, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        sr, sr_seq_len = pad_sequence(sr, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        sent_seq_len = [x + y + 1 for x, y in zip(sl_seq_len, sr_seq_len)]
        desc, desc_seq_len = pad_sequence(desc, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
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
            if self.config.use_pretrained:
                _word_emb = tf.Variable(self.config.emb, name='_word_emb', dtype=tf.float32,
                                        trainable=self.config.finetune_emb)
            else:
                _word_emb = tf.get_variable(name='_word_emb', shape=[self.config.vocab_size, self.config.word_dim],
                                            dtype=tf.float32)
            self.sl_emb = tf.nn.embedding_lookup(_word_emb, self.sl, name='sent_emb')
            self.sr_emb = tf.nn.embedding_lookup(_word_emb, self.sr, name='sent_right_emb')
            self.desc_emb = tf.nn.embedding_lookup(_word_emb, self.desc, name='desc_emb')

        with tf.variable_scope('lexical_encoder'):
            lexical_rnn = bidirectional_dynamic_rnn(self.config.num_units, use_peepholes=True, scope='lexical_rnn')
            self.de = lexical_rnn(self.desc_emb, self.desc_seq_len, return_last_state=True, keep_prob=self.keep_prob,
                                  is_train=self.is_train)  # (batch_size, 2 * num_units)

        with tf.variable_scope('concat_sentence'):
            de = tf.expand_dims(self.de, axis=1)
            s_emb = tf.concat([self.sl_emb, de, self.sr_emb], axis=1)  # (batch_size, seq_len, word_dim)

        with tf.variable_scope('context_encoder'):
            context_rnn = bidirectional_dynamic_rnn(self.config.num_units, use_peepholes=True, scope='context_rnn')
            self.hi = context_rnn(s_emb, self.sent_seq_len, return_last_state=True, keep_prob=self.keep_prob,
                                  is_train=self.is_train)  # (batch_size, 2 * num_units)

        with tf.variable_scope('project'):
            w = tf.get_variable(name='W', shape=[2 * self.config.num_units, 2 * self.config.num_units],
                                dtype=tf.float32)
            b = tf.get_variable("b", shape=[2 * self.config.num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.))
            tmp = tf.nn.bias_add(tf.matmul(self.hi, w), b)  # (batch_size, 2 * num_units)
            tmp = tf.multiply(tmp, self.de)  # (batch_size, 2 * num_units)
            self.logits = tf.sigmoid(tf.reduce_sum(tmp, axis=-1))  # (batch_size, 1)
            # self.logits = dense(tmp, hidden_dim=2, use_bias=True, scope='compute_logits')

    def _build_loss_op(self):
        # labels = tf.one_hot(self.y, depth=2, dtype=tf.float32)
        # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)
        y = tf.cast(self.y, tf.float32)
        losses = -(y * tf.log(self.logits) + (1 - y) * tf.log(1 - self.logits))
        self.loss = tf.reduce_mean(losses)

    def _build_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.config.grad_clip is not None:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grands, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(self.loss)

    '''def _compute_accuracy(self):
        correct_preds = tf.equal(tf.argmax(self.logits, axis=-1), tf.argmax(self.y, axis=-1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32))'''

    def train(self, dataset, devset, sub_set, epochs):
        self.config.logger.info('Start training...')
        nbatches = (len(dataset) * 3 + self.config.batch_size - 1) // self.config.batch_size
        for epoch in range(1, epochs + 1):
            self.config.logger.info('Epoch %2d/%2d:' % (epoch, epochs))
            prog = Progbar(target=nbatches)  # nbatches
            for i, (sl, sr, desc, cand, y) in enumerate(batch_iter(dataset, self.config.batch_size)):
                feed_dict = self._get_feed_dict(sl, sr, desc, cand, True, y=y, lr=self.config.lr,
                                                keep_prob=self.config.keep_prob)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                prog.update(i + 1, [("train loss", train_loss)])
                if (i + 1) % 10 == 0:
                    self.evaluate(sub_set)
            self.config.lr *= self.config.lr_decay
            # build evaluate
            self.evaluate(devset)

    '''def evaluate(self, dataset, batch_size):
        nbatches = (len(dataset) + batch_size - 1) // batch_size
        acc = []
        for sl, sr, desc, cand, y in batch_iter(dataset, batch_size):
            feed_dict = self._get_feed_dict(sl, sr, desc, cand, False, y=y, lr=self.config.lr, keep_prob=1.0)
            batch_acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
            acc.append(batch_acc)
        self.config.logger.info('\nAccuracy: {:04.2f}'.format(sum(acc) / nbatches * 100))'''

    def evaluate(self, dataset):
        acc = []
        for sl, sr, desc, cand, y in batch_iter(dataset, self.config.num_cands):
            feed_dict = self._get_feed_dict(sl, sr, desc, cand, False, y=None, lr=None, keep_prob=1.0)
            prob = self.sess.run(self.logits, feed_dict=feed_dict)
            print(prob, np.argmax(prob), y, np.argmax(y))
            if np.argmax(prob) == np.argmax(y):
                acc.append(1.0)
            else:
                acc.append(0.0)
        self.config.logger.info('\nAccuracy: {:04.2f}'.format(sum(acc) / len(acc) * 100.0))
