from model.doub_enc import DoubEnc
from data.data_prepro import load_json

num_units = 300
lr = 0.001
grad_clip = 5.0
finetune_emb = True
ckpt_path = 'ckpt/'
model_name = 'doub_enc'
embedding_path = 'data/glove.840B.300d.filtered.npz'
batch_size = 32
epochs = 5

print("Loading data...")
train_set = load_json('data/train.json')
dev_set = load_json('data/dev.json')
print(len(train_set))
print(type(train_set[0]))
print(train_set[0])

print("Building model...")
model = DoubEnc(num_units, lr, batch_size, grad_clip, finetune_emb, ckpt_path, embedding_path, model_name)
model.train(train_set, dev_set, batch_size, epochs)
