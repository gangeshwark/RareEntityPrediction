import numpy as np
from dataset.build_vocab_emb import UNK, NUM, PAD, is_digit, load_vocab, load_json
from tqdm import tqdm
from nltk import word_tokenize

word_vocab, _ = load_vocab('../dataset/vocab.txt')
char_vocab, _ = load_vocab('../dataset/chars.txt')
name_desc = load_json('../dataset/name_desc.json')

blank_index = word_vocab['_BLANK_']


def build_train_dev_test_set(dataset, name, use_supplementary=False):
    for record in tqdm(dataset, desc='creating {} set'.format(name)):
        sent1_words = sentence_to_index(record['s1'])
        if use_supplementary:
            sent2_words = sentence_to_index(record['s2'])
        candidates = []
        for can in record['c_ans']:
            can_words = sentence_to_index(name_desc[can])
            candidates.append(can_words)
        # TODO


def sentence_to_index(sentence, use_char=True):
    words = []
    for word in word_tokenize(sentence.strip()):
        word = word_to_index(word, use_char)
        words += [word]
    return words


def word_to_index(word, use_char=True):
    """Convert word into word index, and char index array if use char embeddings"""
    char_ids = []
    if use_char:
        for char in word:
            if char in char_vocab:
                char_ids += [char_vocab[char]]
            else:
                char_ids += [char_vocab[PAD]]  # out of vocab char is represented by PAD token
    if is_digit(word):
        word = NUM
    if word in word_vocab:
        word = word_vocab[word]
    else:
        word = word_vocab[UNK]
    if use_char:
        return char_ids, word
    else:
        return word


def _pad_sequence(sequences, pad_tok, max_length):
    """Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequence(sequences, max_length=None, pad_tok=None, max_length_word=None, nlevels=1):
    """Args:
        sequences: a generator of list or tuple
        max_length: maximal length for a sentence allowed
        max_length_word: maximal length for a word allow, only for nLevels=2
        pad_tok: the char to pad with
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
        sequence_padded, sequence_length = _pad_sequence(sequences, pad_tok, max_length)

    elif nlevels == 2:  # pad chars
        if max_length_word is None:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            seq_padded, seq_length = _pad_sequence(seq, pad_tok, max_length_word)
            sequence_padded += [seq_padded]
            sequence_length += [seq_length]
        if max_length is None:
            max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequence(sequence_padded, [pad_tok] * max_length_word, max_length)
        sequence_length, _ = _pad_sequence(sequence_length, 0, max_length)

    return sequence_padded, sequence_length


def load_embeddings(filename):
    """load word embeddings"""
    try:
        with np.load(filename) as data:
            return data['embeddings']
    except IOError:
        raise "ERROR: Unable to locate file {}.".format(filename)


def main():  # TODO
    dataset = load_json('../dataset/all_data.json')
    train_ratio = 0.8
    dev_ratio = 0.9


if __name__ == '__main__':
    main()
