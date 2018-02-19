from model.model_new import Model
from model.config import Config
from model.utils import load_json

config = Config()

enc = Model(config)

# train_set = load_json('dataset/train.json')
# dev_set = load_json('dataset/dev.json')
# sub_set = dev_set[:config.batch_size]
#
# enc.train(train_set, dev_set, sub_set, epochs=30)
