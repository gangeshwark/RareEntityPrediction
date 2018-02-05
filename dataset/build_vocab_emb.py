import numpy as np
import json
import re
import unicodedata
from nltk import word_tokenize
from tqdm import tqdm
import os
import sys

UNK = '_UNK_'
NUM = '_NUM_'
PAD = '_PAD_'


def is_digit(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    res = re.compile(r'^[-+]?[0-9,]+$').match(s)
    if res:
        return True
    return False


def build_corpus_vocab(corpus_path, use_supplementary=False):
    vocab = set()
    chars_vocab = set()
    dataset = load_vocab(corpus_path)
    for record in tqdm(dataset, desc='build corpus vocabulary'):
        sent1 = record['s1']
        sent2 = record['s2'] if use_supplementary else None
        words1 = word_tokenize(sent1.strip())

        # update char vocab
        for word in words1:
            chars_vocab.update(word)

        words1 = [NUM if is_digit(word) else word for word in words1]
        vocab.update(words1)
        for word in words1:
            chars_vocab.update(word)
        if sent2 is not None:
            words2 = word_tokenize(sent2.strip())

            # update char vocab
            for word in words2:
                chars_vocab.update(word)

            words2 = [NUM if is_digit(word) else word for word in words2]
            vocab.update(words2)
    return vocab, chars_vocab


def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except IOError:
        raise "ERROR: Unable to locate file {}".format(filename)


def build_description_vocab(desc_path):
    vocab = set()
    chars_vocab = set()
    dataset = load_json(desc_path)
    for key, value in tqdm(dataset.items(), desc='build description vocabulary'):
        words = word_tokenize(value.strip())

        # update char vocab
        for word in words:
            chars_vocab.update(word)

        words = [NUM if is_digit(word) else word for word in words]
        vocab.update(words)
    return vocab, chars_vocab


def build_char_vocab(vocab):
    chars_vocab = set()
    for word in vocab:
        chars_vocab.update(word)
    return chars_vocab


def build_embeddings(vocab, glove_path, save_path, dim):
    sys.stdout.write('Creating {} dimension embeddings for vocabulary...'.format(dim))
    scale = np.sqrt(3.0 / dim)
    embeddings = np.random.uniform(-scale, scale, [len(vocab), dim])  # random initialized
    embeddings[0] = np.zeros([1, dim])  # zero vector for PAD, i.e., embeddings[0] = zero vector
    with open(glove_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            word = tokens[0]
            emb = [float(x) for x in tokens[1:]]
            if word in vocab:
                idx = vocab[word]
                embeddings[idx] = np.asarray(emb)
    sys.stdout.write(' done. Saving...')
    np.savez_compressed(save_path, embeddings=embeddings)
    sys.stdout.write(' done.')


def load_vocab(vocab_path):
    try:
        word_idx = dict()
        idx_word = dict()
        with open(vocab_path, 'r') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                word_idx[word] = idx
                idx_word[idx] = word
    except IOError:
        raise 'ERROR: Unable to locate file {}'.format(vocab_path)
    return word_idx, idx_word


def load_glove_vocab(glove_path):
    with open(glove_path, 'r') as f:
        vocab = {line.strip().split()[0] for line in f}
    return vocab


def save_vocab(vocab, file_path):
    with open(file_path, 'w') as f:
        for idx, word in enumerate(vocab):
            if idx != len(vocab) - 1:
                f.write('{}\n'.format(word))
            else:
                f.write(word)


def main():
    # paths
    home = os.path.expanduser('~')
    dataset_dir = '../dataset'
    corpus_path = os.path.join(dataset_dir, 'all_data_unclean.json')
    desc_path = os.path.join(dataset_dir, 'name_desc.json')
    glove_path = os.path.join(home, 'data', 'glove', 'glove.840B.300d.txt')

    # build word and chars vocab for corpus and descriptions
    corpus_vocab, corpus_chars_vocab = build_corpus_vocab(corpus_path, use_supplementary=False)
    desc_vocab, desc_chars_vocab = build_description_vocab(desc_path)

    # merge two chars vocabs and save
    chars_vocab = desc_chars_vocab | corpus_chars_vocab
    chars_vocab = [PAD] + list(chars_vocab)
    save_vocab(chars_vocab, os.path.join(dataset_dir, 'chars.txt'))

    # merge two vocabs
    vocab_words = corpus_vocab | desc_vocab  # union
    print('vocabulary size from dataset: {}'.format(len(vocab_words)))

    # find common vocabulary of dataset and glove
    vocab_glove = load_glove_vocab(glove_path)
    vocab = vocab_words & vocab_glove
    print('common vocabulary size with glove: {}'.format(len(vocab)))

    # add PAD, UNK and NUM tokens to vocab and save
    vocab = [PAD, UNK, NUM] + list(vocab)
    save_vocab(vocab, os.path.join(dataset_dir, 'vocab.txt'))

    # create embeddings
    word_idx, _ = load_vocab(os.path.join(dataset_dir, 'vocab.txt'))
    build_embeddings(word_idx, glove_path, os.path.join(dataset_dir, 'glove.840B.300d.filtered.npz'), 300)


if __name__ == '__main__':
    main()
