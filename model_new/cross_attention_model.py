import tensorflow as tf
from tqdm import tqdm

from dataset.data_prepro import max_sent_len
from model_new.utils import pad_sequence, batch_iter
from model.nns import bidirectional_dynamic_rnn
from model_new.logger import Progbar
import numpy as np
import os


class CrossAttentionModel(object):
    def __init__(self, config):
        self.config = config
        self._add_placeholders()
        self._build_model_op()
        self._build_loss_op()
        self._build_train_op()
        '''self._compute_accuracy()'''
        self.sess, self.saver = None, None
        self.resume_training = True
        self.initialize_session()

    def initialize_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())
        if self.resume_training:
            ch = tf.train.get_checkpoint_state(self.config.ckpt_path)
            print(ch)
            if not ch:
                r = input(
                    "No checkpoint found in directory %s. Can't resume training. Do you want to start a new training session?\n(y)es | (n)o : " % (
                        self.config.ckpt_path))
                if r.startswith('y'):
                    return
                else:
                    exit(0)
            print("Resuming training...")
            print(self.config.model_save_path)
            ckpt_path = ch.model_checkpoint_path
            self.start_epoch = int(ckpt_path.split('-')[-1]) + 1
            print("Start Epoch: ", self.start_epoch)
            self.saver.restore(self.sess, ckpt_path)

    def save_session(self, epoch):
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        self.saver.save(self.sess, self.config.model_save_path, global_step=epoch)

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
        self.batch_sz = self.config.batch_size
        # shape = (batch_size, max length of sentence in batch)
        self.sl = tf.placeholder(tf.int32, shape=[None, max_sent_len], name='sl')
        self.sr = tf.placeholder(tf.int32, shape=[None, max_sent_len], name='sr')
        # shape = (batch_size)
        self.sent_seq_len = tf.placeholder(tf.int32, shape=[None], name='sent_seq_length')
        # shape = (batch_size, max length of sentence in batch)
        self.desc_c1 = tf.placeholder(tf.int32, shape=[None, max_sent_len], name='desc_c1')
        self.desc_c2 = tf.placeholder(tf.int32, shape=[None, max_sent_len], name='desc_c2')
        self.desc_c3 = tf.placeholder(tf.int32, shape=[None, max_sent_len], name='desc_c3')
        # shape = (batch_size)
        self.desc_seq_len_c1 = tf.placeholder(tf.int32, shape=[None], name='desc_seq_len_c1')
        self.desc_seq_len_c2 = tf.placeholder(tf.int32, shape=[None], name='desc_seq_len_c2')
        self.desc_seq_len_c3 = tf.placeholder(tf.int32, shape=[None], name='desc_seq_len_c3')
        # shape = (batch_size)
        self.cand = tf.placeholder(tf.int32, shape=[None, self.config.num_cands], name='candidates')
        # shape = (batch_size)
        self.y = tf.placeholder(tf.int32, shape=[None, self.config.num_cands], name='answer')
        # hyperparameters
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.lr = tf.placeholder(dtype=tf.float32, name='lr')
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')

    def _get_feed_dict(self, sl, sr, desc1, desc2, desc3, cand, is_train, y=None, lr=None,
                       keep_prob=None, batch_size=None):
        sl, sl_seq_len = pad_sequence(sl, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        sr, sr_seq_len = pad_sequence(sr, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        sent_seq_len = [x + y + 1 for x, y in zip(sl_seq_len, sr_seq_len)]
        desc1, desc_seq_len1 = pad_sequence(desc1, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        desc2, desc_seq_len2 = pad_sequence(desc2, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        desc3, desc_seq_len3 = pad_sequence(desc3, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        feed_dict = {
            self.sl: sl, self.sr: sr, self.sent_seq_len: sent_seq_len,
            self.desc_c1: desc1, self.desc_seq_len_c1: desc_seq_len1,
            self.desc_c2: desc2, self.desc_seq_len_c2: desc_seq_len2,
            self.desc_c3: desc3, self.desc_seq_len_c3: desc_seq_len3,
            self.cand: cand,
            self.is_train: is_train}
        if y is not None:
            feed_dict[self.y] = y
        if lr is not None:
            feed_dict[self.lr] = lr
        if keep_prob is not None:
            feed_dict[self.keep_prob] = keep_prob
        if batch_size is not None:
            feed_dict[self.batch_sz] = batch_size
        return feed_dict

    def _build_model_op(self):
        with tf.variable_scope('embeddings'):
            _word_emb = tf.Variable(self.config.emb, name='_word_emb', dtype=tf.float32,
                                    trainable=self.config.finetune_emb)
            self.sl_emb = tf.nn.embedding_lookup(_word_emb, self.sl, name='sent_left_emb')
            self.sr_emb = tf.nn.embedding_lookup(_word_emb, self.sr, name='sent_right_emb')
            self.desc_emb_c1 = tf.nn.embedding_lookup(_word_emb, self.desc_c1, name='desc_emb')
            self.desc_emb_c2 = tf.nn.embedding_lookup(_word_emb, self.desc_c2, name='desc_emb')
            self.desc_emb_c3 = tf.nn.embedding_lookup(_word_emb, self.desc_c3, name='desc_emb')

        with tf.variable_scope('lexical_encoder'):
            lexical_rnn = bidirectional_dynamic_rnn(self.config.num_units, use_peepholes=True, scope='lexical_rnn')
            self.de1 = lexical_rnn(self.desc_emb_c1, self.desc_seq_len_c1, return_last_state=True,
                                   keep_prob=self.keep_prob,
                                   is_train=self.is_train)  # (batch_size, 2 * num_units)
            self.de2 = lexical_rnn(self.desc_emb_c2, self.desc_seq_len_c2, return_last_state=True,
                                   keep_prob=self.keep_prob,
                                   is_train=self.is_train)  # (batch_size, 2 * num_units)
            self.de3 = lexical_rnn(self.desc_emb_c3, self.desc_seq_len_c3, return_last_state=True,
                                   keep_prob=self.keep_prob,
                                   is_train=self.is_train)  # (batch_size, 2 * num_units)

        with tf.variable_scope('concat_sentence'):
            de1 = tf.expand_dims(self.de1, axis=1)
            de2 = tf.expand_dims(self.de2, axis=1)
            de3 = tf.expand_dims(self.de3, axis=1)
            s_emb1 = tf.concat([self.sl_emb, de1, self.sr_emb], axis=1)  # (batch_size, seq_len, word_dim)
            s_emb2 = tf.concat([self.sl_emb, de2, self.sr_emb], axis=1)  # (batch_size, seq_len, word_dim)
            s_emb3 = tf.concat([self.sl_emb, de3, self.sr_emb], axis=1)  # (batch_size, seq_len, word_dim)

        with tf.variable_scope('context_encoder'):
            context_rnn = bidirectional_dynamic_rnn(self.config.num_units, use_peepholes=True, scope='context_rnn')
            self.hi_c1 = context_rnn(s_emb1, self.sent_seq_len, return_last_state=True, keep_prob=self.keep_prob,
                                     is_train=self.is_train)  # (batch_size, 2 * num_units)
            self.hi_c2 = context_rnn(s_emb2, self.sent_seq_len, return_last_state=True, keep_prob=self.keep_prob,
                                     is_train=self.is_train)  # (batch_size, 2 * num_units)
            self.hi_c3 = context_rnn(s_emb3, self.sent_seq_len, return_last_state=True, keep_prob=self.keep_prob,
                                     is_train=self.is_train)  # (batch_size, 2 * num_units)

        """
        with tf.variable_scope('project'):
            w = tf.get_variable(name='W', shape=[2 * self.config.num_units, 2 * self.config.num_units],
                                dtype=tf.float32)
            b = tf.get_variable("b", shape=[2 * self.config.num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.))
            tmp = tf.nn.bias_add(tf.matmul(self.hi, w), b)  # (batch_size, 2 * num_units)
            tmp = tf.multiply(tmp, self.de)  # (batch_size, 2 * num_units)
            self.logits = tf.sigmoid(tf.reduce_sum(tmp, axis=-1))  # (batch_size, 1)
            # self.logits = dense(tmp, hidden_dim=2, use_bias=True, scope='compute_logits')
        """
        with tf.variable_scope('project'):
            W = tf.get_variable(name='W', shape=[2 * self.config.num_units, 2 * self.config.num_units],
                                dtype=tf.float32)
            b = tf.get_variable("b", shape=[2 * self.config.num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.))
            tmp_c1 = tf.nn.bias_add(tf.matmul(self.hi_c1, W), b)  # (batch_size, 2 * num_units)
            tmp_c2 = tf.nn.bias_add(tf.matmul(self.hi_c2, W), b)
            tmp_c3 = tf.nn.bias_add(tf.matmul(self.hi_c3, W), b)
            tmp_c1 = tf.reduce_sum(tf.matmul(tmp_c1, tf.transpose(self.de1)), axis=-1)  # (batch_size, 1)
            tmp_c2 = tf.reduce_sum(tf.matmul(tmp_c2, tf.transpose(self.de2)), axis=-1)
            tmp_c3 = tf.reduce_sum(tf.matmul(tmp_c3, tf.transpose(self.de3)), axis=-1)
            tmp_c1 = tf.reshape(tmp_c1, shape=[-1, 1])  # (batch_size, 1)
            tmp_c2 = tf.reshape(tmp_c2, shape=[-1, 1])  # (batch_size, 1)
            tmp_c3 = tf.reshape(tmp_c3, shape=[-1, 1])  # (batch_size, 1)

            self.logits = tf.concat([tf.sigmoid(tmp_c1), tf.sigmoid(tmp_c2), tf.sigmoid(tmp_c3)],
                                    axis=1)  # (batch_size, 3)
            # self.logits = tf.nn.softmax(self.logits1)

    def _build_loss_op(self):
        # labels = tf.one_hot(self.y, depth=2, dtype=tf.float32)
        y = tf.cast(self.y, tf.float32)
        # y = tf.argmax(y, axis=-1)
        # print("y shape: ", y.get_shape())
        # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y)
        # losses = tf.losses.log_loss(predictions=self.logits, labels=y)
        # losses = -(y * tf.log(self.logits) + (1 - y) * tf.log(1 - self.logits))
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
        nbatches = (len(dataset)) // self.batch_sz
        for epoch in range(1, epochs + 1):
            self.config.logger.info('Epoch %2d/%2d:' % (epoch, epochs))
            prog = Progbar(target=nbatches)  # nbatches
            for i, (sl, sr, desc1, desc2, desc3, cand, y) in enumerate(batch_iter(dataset, self.batch_sz)):
                feed_dict = self._get_feed_dict(sl, sr, desc1, desc2, desc3, cand, True, y=y, lr=self.config.lr,
                                                keep_prob=self.config.keep_prob)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                prog.update(i + 1, [("train loss", train_loss)])
                if (i + 1) % 1000 == 0:
                    self.evaluate(sub_set, epoch, i + 1, train_loss)
            # self.config.lr *= self.config.lr_decay
            # build evaluate
            self.evaluate(devset, epoch)
            self.save_session(epoch)

    '''def evaluate(self, dataset, batch_size):
        nbatches = (len(dataset) + batch_size - 1) // batch_size
        acc = []
        for sl, sr, desc, cand, y in batch_iter(dataset, batch_size):
            feed_dict = self._get_feed_dict(sl, sr, desc, cand, False, y=y, lr=self.config.lr, keep_prob=1.0)
            batch_acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
            acc.append(batch_acc)
        self.config.logger.info('\nAccuracy: {:04.2f}'.format(sum(acc) / nbatches * 100))'''

    def evaluate(self, dataset, epoch, step=None, loss=None):
        acc = []
        print("\n")
        # print(len(dataset), '\n')
        for sl, sr, desc1, desc2, desc3, cand, y in tqdm(batch_iter(dataset, self.batch_sz)):
            feed_dict = self._get_feed_dict(sl, sr, desc1, desc2, desc3, cand, False, y=None, lr=None, keep_prob=1.0)
            prob = self.sess.run(self.logits, feed_dict=feed_dict)
            # print(prob, np.argmax(prob, axis=-1), (np.argmax(prob, axis=-1)).shape, y, np.argmax(y, axis=-1),
            #       (np.argmax(y, axis=-1)).shape)
            a = np.argmax(prob, axis=-1) == np.argmax(y, axis=-1)
            a = a.astype(np.int32)
            acc.extend(a)
            # np.sum()
        accuracy = sum(acc) / len(acc) * 100.0
        # print(acc)
        if not step:
            step = -1
        if loss:
            self.config.logger.info(
                '\n@Epoch {} Step {} -> Accuracy: {:04.2f} - Train loss: {} '.format(epoch, step, accuracy, loss))
        else:
            self.config.logger.info('\n@Epoch {} & Step {} -> Accuracy: {:04.2f}'.format(epoch, step, accuracy))
