import numpy as np
import keras
import keras.backend as K
from keras import Input, Model
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Concatenate, Activation, Dot, RepeatVector

from dataset.data_prepro_new import max_sent_len, load_json
from model_new.config import Config
from model_new.utils import pad_sequence

name_desc = load_json('dataset/name_desc.json')


def batch_iter(dataset):
    batch_sl, batch_sr, batch_desc1, batch_desc2, batch_desc3, batch_cand, batch_y = [], [], [], [], [], [], []
    for record in dataset:
        batch_sl += [record['cl']]
        batch_sr += [record['cr']]
        desc = [name_desc[str(can)] for can in record['c_ans']]
        batch_desc1 += [desc[0]]
        batch_desc2 += [desc[1]]
        batch_desc3 += [desc[2]]
        batch_cand += [record['c_ans']]
        y = [1 if x == record['ans'] else 0 for x in record['c_ans']]
        batch_y += [y]

    return batch_sl, batch_sr, batch_desc1, batch_desc2, batch_desc3, batch_cand, batch_y
    # if len(batch_sl) != 0:
    #     yield batch_sl, batch_sr, batch_desc1, batch_desc2, batch_desc3, batch_cand, batch_y


class KerasModel():
    def __init__(self, config):
        self.config = config
        self.batch_sz = self.config.batch_size
        self._build_model()

    def _build_model(self):
        self.sl = Input(shape=(max_sent_len,), name='sent_left')
        self.sr = Input(shape=(max_sent_len,), name='sent_right')
        self.sent_seq_len = Input(shape=(1,), name='sent_seq_length')
        # shape = (batch_size, max length of sentence in batch)
        self.desc_c1 = Input(shape=(max_sent_len,), name='desc_c1')
        self.desc_c2 = Input(shape=(max_sent_len,), name='desc_c2')
        self.desc_c3 = Input(shape=(max_sent_len,), name='desc_c3')
        # shape = (batch_size)
        self.desc_seq_len_c1 = Input(shape=(1,), name='desc_seq_len_c1')
        self.desc_seq_len_c2 = Input(shape=(1,), name='desc_seq_len_c2')
        self.desc_seq_len_c3 = Input(shape=(1,), name='desc_seq_len_c3')
        # shape = (batch_size)
        self.cand = Input(shape=(self.config.num_cands,), name='candidates')
        # shape = (batch_size)
        self.y = Input(shape=(self.config.num_cands,), name='answer')
        # hyperparameters
        self.keep_prob = self.config.keep_prob
        self.lr = self.config.lr
        self.is_train = True
        self.vocab_length = 408010

        embedding_layer = Embedding(self.vocab_length,
                                    self.config.word_dim,
                                    weights=[self.config.emb],
                                    input_length=max_sent_len,
                                    trainable=False)

        lexical_encoder = LSTM(300, kernel_initializer='uniform', return_state=True,
                               input_shape=(max_sent_len, self.config.word_dim), name='lexical_encoder')
        context_encoder = LSTM(300, kernel_initializer='uniform', return_state=True,
                               input_shape=(max_sent_len + 1 + max_sent_len, self.config.word_dim),
                               name='context_encoder')
        dense = Dense(300, use_bias=True)
        concat = Concatenate(axis=1)

        final_concat = Concatenate(axis=1)
        sigmoid = Activation('sigmoid')
        dot = Dot(-1)
        repeat = RepeatVector(1)

        self.sl_emb = embedding_layer(self.sl)
        self.sr_emb = embedding_layer(self.sr)
        self.desc_emb_c1 = embedding_layer(self.desc_c1)
        self.desc_emb_c2 = embedding_layer(self.desc_c2)
        self.desc_emb_c3 = embedding_layer(self.desc_c3)
        print(self.desc_emb_c1._keras_shape)

        _, self.de1, _ = lexical_encoder([self.desc_emb_c1])
        _, self.de2, _ = lexical_encoder([self.desc_emb_c2])
        _, self.de3, _ = lexical_encoder([self.desc_emb_c3])
        de1 = repeat(self.de1)
        de2 = repeat(self.de2)
        de3 = repeat(self.de3)
        print("de1._keras_shape", de1._keras_shape)
        # print(self.de2.get_shape())
        # print(self.de3.get_shape())
        s_emb1 = concat([self.sl_emb, de1, self.sr_emb])
        s_emb2 = concat([self.sl_emb, de2, self.sr_emb])
        s_emb3 = concat([self.sl_emb, de3, self.sr_emb])
        print(s_emb1._keras_shape)
        _, self.hi_c1, _ = context_encoder(s_emb1)
        _, self.hi_c2, _ = context_encoder(s_emb2)
        _, self.hi_c3, _ = context_encoder(s_emb3)
        print(self.hi_c1._keras_shape)

        tmp_c1 = dense(self.hi_c1)
        tmp_c2 = dense(self.hi_c2)
        tmp_c3 = dense(self.hi_c3)
        tmp_c1 = dot([tmp_c1, self.de1])
        tmp_c2 = dot([tmp_c2, self.de2])
        tmp_c3 = dot([tmp_c3, self.de3])
        print(tmp_c1.get_shape())
        print(tmp_c2.get_shape())
        print(tmp_c3.get_shape())
        o1 = sigmoid(tmp_c1)
        o2 = sigmoid(tmp_c2)
        o3 = sigmoid(tmp_c3)
        print(o1.get_shape())
        pred = final_concat([o1, o2, o3])
        self.model = Model(
            inputs=[self.sl, self.sr, self.sent_seq_len, self.desc_c1, self.desc_c2, self.desc_c3, self.desc_seq_len_c1,
                    self.desc_seq_len_c2, self.desc_seq_len_c3, self.cand], outputs=pred)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train(self, dataset, devset, sub_set):
        sl, sr, desc1, desc2, desc3, cand, y = batch_iter(dataset)
        sl, sl_seq_len = pad_sequence(sl, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        sr, sr_seq_len = pad_sequence(sr, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        sent_seq_len = [x + y + 1 for x, y in zip(sl_seq_len, sr_seq_len)]
        desc1, desc_seq_len1 = pad_sequence(desc1, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        desc2, desc_seq_len2 = pad_sequence(desc2, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)
        desc3, desc_seq_len3 = pad_sequence(desc3, max_length=max_sent_len, pad_tok=0, pad_left=True, nlevels=1)

        sl = np.array(sl)
        sr = np.array(sr)
        sent_seq_len = np.array(sent_seq_len)
        desc1 = np.array(desc1)
        desc2 = np.array(desc2)
        desc3 = np.array(desc3)
        desc_seq_len1 = np.array(desc_seq_len1)
        desc_seq_len2 = np.array(desc_seq_len2)
        desc_seq_len3 = np.array(desc_seq_len3)
        cand = np.array(cand)

        self.model.fit(x=[sl, sr, sent_seq_len, desc1, desc2, desc3, desc_seq_len1, desc_seq_len2, desc_seq_len3, cand],
                       y=y, validation_split=0.1, batch_size=64, epochs=30, verbose=2)
