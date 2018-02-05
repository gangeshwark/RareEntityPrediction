from model.doub_enc import DoubEnc
from data.data_prepro import load_json
from model.utils import batch_iter

num_units = 300
lr = 0.001
grad_clip = 5.0
finetune_emb = True
ckpt_path = 'ckpt/'
model_name = 'doub_enc'
embedding_path = 'data_new/glove.840B.300d.filtered.npz'
batch_size = 32
epochs = 5

print("Loading data...")
train_set = load_json('data_new2/train.json')
dev_set = load_json('data_new2/dev.json')
print(len(train_set))
print(type(train_set[0]))
# print(train_set[0])
s1, _, idx, desc, cand, y = batch_iter(train_set, 2)
print(s1)
print(idx)
print(desc)
print(y)
print(cand)
