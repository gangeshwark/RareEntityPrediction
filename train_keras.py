from model_new.config import Config
from model_new.keras_model import KerasModel
from model_new.utils import load_json

config = Config()
model = KerasModel(config)

train_set = load_json('dataset/train.json')
# dev_set = load_json('dataset/dev.json')
# sub_set = dev_set[:config.batch_size * 50]

model.train(train_set, None, None)
