from model_new.models import Model
from model_new.config import Config
from model_new.utils import load_json

config = Config()
print(len(config.emb), type(config.emb))
print(config.emb.shape)

enc = Model(config)
#
train_set = load_json('dataset/train.json')
dev_set = load_json('dataset/dev.json')
sub_set = dev_set[:config.batch_size * 50]

enc.train(train_set, dev_set, sub_set, epochs=30)
