from model.model_new import Model
from model.config import Config
from model.utils_new import load_json

config = Config()
config.batch_size = 5
enc = Model(config)

train_set = load_json('dataset/train.json')
dev_set = load_json('dataset/dev.json')
sub_set = dev_set[:config.batch_size]

enc.debug(train_set, dev_set, sub_set, epochs=1)
