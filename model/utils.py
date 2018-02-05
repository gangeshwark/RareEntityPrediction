import numpy as np
from data.data_prepro import load_json, load_vocab, max_sent_len, max_sent_len_desc


def batch_iter(dataset, batch_size):
    name_desc = load_json('data/name_desc.json')
    batch_s1, batch_s2, batch_idx, batch_desc, batch_cand, batch_y = [], [], [], [], [], []
    for record in dataset:
        if len(batch_s1) == batch_size:
            yield batch_s1, batch_s2, batch_idx, batch_desc, batch_cand, batch_y
            batch_s1, batch_s2, batch_idx, batch_desc, batch_cand, batch_y = [], [], [], [], [], []
        batch_s1 += [record['s1']]
        batch_s2 += [record['s2']]
        batch_idx += [record['idx']]
        desc = [name_desc[str(can)] for can in record['c_ans']]
        # print("DESC:", desc)
        batch_desc += [desc]
        batch_cand += [record['c_ans']]
        y = []
        for x in record['c_ans']:
            if x == record['ans']:
                y.append(1)
            else:
                y.append(0)
        batch_y += [y]
    if len(batch_s1) != 0:
        yield batch_s1, batch_s2, batch_idx, batch_desc, batch_cand, batch_y


def load_embeddings(filename):
    """load word embeddings"""
    try:
        with np.load(filename) as data:
            return data['embeddings']
    except IOError:
        raise "ERROR: Unable to locate file {}.".format(filename)


def _pad_sequence(sequences, pad_tok, max_length, pad_left):
    """Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        if pad_left:
            seq_ = [pad_tok] * max(max_length - len(seq), 0) + seq[:max_length]
        else:
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequence(sequences, max_length=None, pad_tok=None, max_length_2=None, pad_left=True, nlevels=1):
    """Args:
        sequences: a generator of list or tuple
        max_length: maximal length for a sentence allowed
        max_length_2: maximal length for a word allow, only for nLevels=2
        pad_tok: the char to pad with
        pad_left: pad from left side
        nlevels: "depth" of padding, 2 for the case where we have characters ids
    Returns:
        a list of list where each sublist has same length
    """
    if pad_tok is None:
        pad_tok = 0
    sequence_padded, sequence_length = [], []
    if nlevels == 1:  # pad words
        if max_length is None:
            max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequence(sequences, pad_tok, max_length, pad_left)

    elif nlevels == 2:  # pad chars
        if max_length_2 is None:
            max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            seq_padded, seq_length = _pad_sequence(seq, pad_tok, max_length_2, pad_left)
            sequence_padded += [seq_padded]
            sequence_length += [seq_length]
        if max_length is None:
            max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequence(sequence_padded, [pad_tok] * max_length_2, max_length, pad_left)
        sequence_length, _ = _pad_sequence(sequence_length, 0, max_length, pad_left)

    return sequence_padded, sequence_length
