from model.utils import pad_sequence
from pprint import pprint

short = [1, 2, 3]
long = [1, 2, 3, 4, 5, 6]

record_1 = [short, long]
record_2 = [short, long]

# test 2 dim data
record_1_pad, seq_lens = pad_sequence(record_1, max_length=6, pad_tok=0, pad_left=True, nlevels=1)
pprint(record_1)
pprint(record_1_pad)
print()

# test 3 dim data
batch = [record_1, record_2]  # 3 dimensional (batch_size, num_cands, sentence)
pprint(batch)
batch_pad, seq_lengths = pad_sequence(batch, max_length=None, max_length_2=7, pad_tok=0, pad_left=True, nlevels=2)
pprint(batch_pad)
print()
pprint(seq_lengths)
