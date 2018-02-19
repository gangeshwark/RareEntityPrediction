import os
import re
import numpy as np
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize
from collections import OrderedDict
import json

np.random.seed(12345)

home = os.path.expanduser('~')
source_dir = os.path.join(home, 'data', 'rare_entity')
dataset_dir = os.path.join('..', 'dataset')

id_finder = re.compile(r'(9202a8c04000641f8\w+)')
special_character = re.compile(r'[^A-Za-z_\d,.\- ]', re.IGNORECASE)

id_replace = '__blank__'
UNK = '_UNK_'
PAD = '_PAD_'

# according to statistical information
max_sent_len = 60


def clean_text(text, lower=False):
    if lower:
        text = text.lower()
    # remove Special Characters
    text = special_character.sub('', text)
    # replace multiple spaces with single one.
    text = re.sub(' +', ' ', text)
    return text


def json_dump(data, file_path):
    """Save data into file with json format"""
    if data is not None:
        with open(file_path, 'w') as f:
            json.dump(data, f)
    print('dump data to {} successful...'.format(file_path))


def prepro_entities(entity_file, lower=False):
    id_name = {}  # freebase id and entity name
    name_desc = {}  # entity name and one sentence description
    word_count = {}
    with open(entity_file, 'r', encoding='utf-8') as fe:
        for line in tqdm(fe, desc='process entity file'):
            fb_id, *_, name, desc = line.strip().split('\t')  # fb_id, anchor_text, wiki_url, fb_name, and description
            # clean additional backspace
            fb_id, name = fb_id.strip(), name.strip()
            desc = sent_tokenize(desc, language='english')[0]  # NLTK sentence tokenization, get first one
            desc = word_tokenize(clean_text(desc, lower))  # clean and tokenize text
            # cut down sentences according to the threshold
            if len(desc) > max_sent_len:
                desc = desc[:max_sent_len]
            # record id and name pair
            id_name[fb_id] = name
            # record name and description pair
            name_desc[name] = desc
            # count word frequency
            for word in desc:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
    return id_name, name_desc, word_count


def prepro_corpus(corpus_file, id_name, lower=False, num_cands=3):
    data = []
    word_count = {}
    names = list(id_name.values())
    num_names = len(names)
    with open(corpus_file, 'r', encoding='utf-8') as fc:
        for _, line in tqdm(enumerate(fc), desc='process corpus file'):
            line = line.strip().replace("``", '"').replace("''", '"')
            all_ids = id_finder.findall(line)
            candidates = [id_name[fb_id] for fb_id in all_ids]
            candidates = list(set(candidates))
            sentences = sent_tokenize(line)  # split paragraph into sentences
            # filtered out sentences too short
            sentences = [sent for sent in sentences if len(sent) >= 10]
            for i in range(len(sentences)):
                if not id_finder.search(sentences[i]):  # if no freebase ids are contained, ignore this sentence
                    continue
                sent_ids = id_finder.findall(sentences[i])  # obtain all freebase ids in a sentence
                for fb_id in sent_ids:
                    ans = id_name[fb_id]
                    other_ids = [x for x in sent_ids if x != fb_id]  # get other freebase ids
                    sent = sentences[i].replace(fb_id, id_replace)  # replace target id with __blank__
                    for sub_id in other_ids:
                        sent = sent.replace(sub_id, id_name[sub_id])  # replace other ids with actual name
                    # clean and tokenize sentence
                    sent = word_tokenize(clean_text(sent, lower))
                    # if contains more than one __blank__ ignore
                    if sent.count(id_replace) > 1:
                        continue

                    # supplementary information
                    pre_sent = None if i == 0 else sentences[i - 1]
                    if pre_sent is not None:
                        for sub_id in id_finder.findall(pre_sent):
                            pre_sent = pre_sent.replace(sub_id, id_name[sub_id])
                        pre_sent = word_tokenize(clean_text(pre_sent, lower))
                        sent = pre_sent + sent
                    nxt_sent = None if i == len(sentences) - 1 else sentences[i + 1]
                    if nxt_sent is not None:
                        for sub_id in id_finder.findall(nxt_sent):
                            nxt_sent = nxt_sent.replace(sub_id, id_name[sub_id])
                        nxt_sent = word_tokenize(clean_text(nxt_sent, lower))
                        sent = sent + nxt_sent

                    # create candidates set
                    cands = []
                    if len(candidates) < num_cands:  # pick candidates from all data
                        cands += candidates
                        for _ in range(num_cands - len(candidates)):
                            rand = np.random.randint(num_names, size=1)[0]
                            while names[rand] in cands:
                                rand = np.random.randint(num_names, size=1)[0]
                            cands += [names[rand]]
                    else:  # pick several from current candidates
                        cands += [ans]
                        tmp = [x for x in candidates if x != ans]
                        cands += tmp[:2]
                    # shuffle candidates
                    assert len(cands) == num_cands
                    np.random.shuffle(cands)

                    blank_idx = sent.index(id_replace)  # __blank__ index
                    # split first sentence into two parts according to the position of __blank__
                    left_cut = max(0, blank_idx - max_sent_len)
                    right_cut = min(len(sent), blank_idx + 1 + max_sent_len)
                    sent_left = sent[left_cut:blank_idx]
                    sent_right = sent[blank_idx + 1:right_cut]

                    # count the words frequency
                    words = sent_left + sent_right
                    for word in words:
                        if word in word_count:
                            word_count[word] += 1
                        else:
                            word_count[word] = 1

                    # store each record into dict
                    record = {"sl": sent_left,
                              "sr": sent_right,
                              "c_ans": cands,  # fix length with num_cands
                              "ans": ans}
                    data.append(record)

    np.random.shuffle(data)
    return data, word_count


def build_vocab(ent_words, cor_words, threshold=5):
    for word, count in ent_words.items():
        if word in cor_words:
            cor_words[word] += count
        else:
            cor_words[word] = count
    cor_words = OrderedDict(sorted(cor_words.items(), key=lambda val: val[1], reverse=True))
    vocab = []
    char_vocab = set()
    for word, count in cor_words.items():
        char_vocab.update(word)
        if count >= threshold:
            vocab.append(word)
        else:
            break
    return vocab, char_vocab


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


def save_vocab(vocab, file_path):
    with open(file_path, 'w') as f:
        for idx, word in enumerate(vocab):
            if idx != len(vocab) - 1:
                f.write('{}\n'.format(word))
            else:
                f.write(word)


def save_json(dataset, save_path):
    if dataset is not None:
        with open(save_path, 'w') as f:
            json.dump(dataset, f)


def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except IOError:
        raise "ERROR: Unable to locate file {}".format(filename)


def build_embeddings(vocab, emb_path, save_path, dim):
    print('Create embeddings...')
    scale = np.sqrt(3.0 / dim)
    embeddings = np.random.uniform(-scale, scale, [len(vocab), dim])  # random initialized
    embeddings[0] = np.zeros([1, dim])  # zero vector for PAD, i.e., embeddings[0] = zero vector
    with open(emb_path, 'r') as f:
        for line in tqdm(f, desc='update embeddings'):
            tokens = line.strip().split(' ')
            word = tokens[0]
            emb = [float(x) for x in tokens[1:]]
            if word in vocab:
                idx = vocab[word]
                embeddings[idx] = np.asarray(emb)
    print(' done. Saving...')
    np.savez_compressed(save_path, embeddings=embeddings)


def convert_to_index(data, vocab, entity_names, save_path, name):
    dataset = []
    for record in tqdm(data, desc='convert {} dataset to index'.format(name)):
        record_idx = {"cl": [vocab[word] if word in vocab else vocab[UNK] for word in record['sl']],
                      "cr": [vocab[word] if word in vocab else vocab[UNK] for word in record['sr']],
                      "c_ans": [entity_names[name] for name in record['c_ans']],
                      "ans": entity_names[record['ans']]}
        dataset.append(record_idx)
    json_dump(dataset, save_path)
    del dataset


def main():
    """File path"""
    # source path
    entities_path = os.path.join(source_dir, 'entities.txt')
    corpus_path = os.path.join(source_dir, 'corpus.txt')
    glove_path = os.path.join(home, 'data', 'glove', 'glove.840B.300d.txt')

    # created data path
    entity_names_path = os.path.join(dataset_dir, 'entity_names.txt')
    word_vocab_path = os.path.join(dataset_dir, 'words.txt')
    char_vocab_path = os.path.join(dataset_dir, 'chars.txt')
    glove_save_path = os.path.join(dataset_dir, 'glove.840B.300d.filtered.npz')
    name_desc_path = os.path.join(dataset_dir, 'name_desc.json')
    train_data_path = os.path.join(dataset_dir, 'train.json')
    dev_data_path = os.path.join(dataset_dir, 'dev.json')
    test_data_path = os.path.join(dataset_dir, 'test.json')

    """Build dataset"""
    id_name, name_desc, ent_vocab = prepro_entities(entities_path)
    data, cor_vocab = prepro_corpus(corpus_path, id_name)
    entity_names = id_name.values()
    save_vocab(entity_names, entity_names_path)
    del id_name  # delete unused items to release space

    """Build and save words and chars vocabulary"""
    print('Creating words and chars vocabularies...')
    vocab, char_vocab = build_vocab(ent_vocab, cor_vocab, threshold=5)
    del ent_vocab, cor_vocab
    vocab = [PAD, UNK] + list(vocab)
    save_vocab(vocab, word_vocab_path)
    char_vocab = [PAD] + list(char_vocab)
    save_vocab(char_vocab, char_vocab_path)
    del vocab, char_vocab

    """Convert all dataset by index"""
    vocab, _ = load_vocab(os.path.join(dataset_dir, 'words.txt'))
    entity_names, _ = load_vocab(entity_names_path)
    # convert name_desc
    name_desc_idx = {}
    for name, desc in tqdm(name_desc.items(), desc='convert name_desc to index'):
        words = [vocab[word] if word in vocab else vocab[UNK] for word in desc]
        name = entity_names[name]
        name_desc_idx[name] = words
    save_json(name_desc_idx, name_desc_path)
    del name_desc, name_desc_idx

    # convert data
    data_size = len(data)
    train_end, dev_end = int(data_size * 0.8), int(data_size * 0.9)
    convert_to_index(data[:train_end], vocab, entity_names, train_data_path, 'train')
    convert_to_index(data[train_end:dev_end], vocab, entity_names, dev_data_path, 'dev')
    convert_to_index(data[dev_end:], vocab, entity_names, test_data_path, 'test')
    del data

    """Build and save word embeddings"""
    build_embeddings(vocab, glove_path, glove_save_path, dim=300)


if __name__ == '__main__':
    main()
