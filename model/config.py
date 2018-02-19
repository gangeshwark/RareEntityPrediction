import os
from model.logger import get_logger
from model.utils import load_embeddings


class Config(object):
    def __init__(self):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.logger = get_logger(self.log_file)
        self.emb = load_embeddings(self.emb_file)
        # self.emb = load_embeddings(self.emb_file)

    __head_dir = 'dataset'
    train_file = os.path.join(__head_dir, 'train.json')
    dev_file = os.path.join(__head_dir, 'dev.json')
    test_file = os.path.join(__head_dir, 'test.json')
    emb_file = os.path.join(__head_dir, 'glove.840B.300d.filtered.npz')
    finetune_emb = True
    word_dim = 300

    num_cands = 3
    batch_size = 32

    ckpt_path = 'ckpt'
    log_file = os.path.join(ckpt_path, 'log.txt')
    model_name = 'bi_doub_enc'

    keep_prob = 0.5
    grad_clip = 5.0
    lr = 0.001
    lr_decay = 0.9

    desc_units = 150
    num_units = 150
