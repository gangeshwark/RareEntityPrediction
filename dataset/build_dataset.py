import os
import re
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize, wordpunct_tokenize, regexp_tokenize
from nltk.corpus import stopwords
import json

home = os.path.expanduser('~')
source_dir = os.path.join(home, 'data', 'rare_entity')

prog = re.compile(r'(9202a8c04000641f8\w+)')
special_character = re.compile(r'[^a-z_\d ]', re.IGNORECASE)

stops = set(stopwords.words("english"))

id_replace = '_BLANK_'


def clean_text(text):
    text = word_tokenize(text.lower())
    # remove stop words.
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    # remove Special Characters
    text = special_character.sub('', text)
    # replace multiple spaces with single one.
    text = re.sub(' +', ' ', text)
    # return the cleaned text
    return text


def prepro_entities(entity_file, dump_id_name=None, dump_name_desc=None):
    id_name = {}
    name_desc = {}
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='process entities'):
            fb_id, *_, name, full_desc = line.strip().split('\t')
            desc = sent_tokenize(full_desc, language='english')[0]  # NLTK sentence tokenization, get first sentence
            # desc = ' '.join([token.replace("``", '"').replace("''", '"') for token in word_tokenize(desc)])

            # clean text
            desc = clean_text(desc)

            id_name[fb_id] = name
            name_desc[name] = desc
    if dump_id_name is not None:
        with open(dump_id_name, 'w') as fi:
            json.dump(id_name, fi)
    if dump_name_desc is not None:
        with open(dump_name_desc, 'w') as fn:
            json.dump(name_desc, fn)
    return id_name, name_desc


def prepro_corpus(corpus_file, id_name):
    data = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='process corpus'):  # loop each document
            all_ids = prog.findall(line)

            if len(all_ids) > 10:  # filter out doc with more than 10 blanks (paper P3 Table 2)
                continue

            candidates = [id_name[fb_id] for fb_id in all_ids]

            line = line.replace("``", '"').replace("''", '"')  # used same quote

            # one method, makes sure the doc is truly split into sentences, some drawbacks, like,
            # high-fat -> high - fat , 'm -> ' m , ca n't -> ca n ' t
            # 5.88 -> 5 . 88 and it will breaks into two sentences
            # Generally, everything will be separated
            # line = ' '.join(wordpunct_tokenize(line))
            # sents = [sent for sent in sent_tokenize(line) if len(sent) >= 10]

            # alternative method (needs to carefully define a regression expression)
            line = ' '.join(regexp_tokenize(line, pattern="\w*?-?\w+-\w+|\w*'\w+|[a-zA-z]+\S+?|\S+"))
            sents = [' '.join(word_tokenize(sent)) for sent in sent_tokenize(line) if len(sent) >= 10]

            for i in range(len(sents)):  # process each sentence
                if not prog.search(sents[i]):  # no freebase ids are contained
                    continue
                ids = prog.findall(sents[i])  # obtain all freebase ids in a sentence
                for fb_id in ids:
                    answer = id_name[fb_id]
                    other_ids = [x for x in ids if x != fb_id]  # get other freebase ids
                    fst_sent = sents[i].replace(fb_id, id_replace)  # replace target id with **BLANK**
                    for sub_id in other_ids:
                        fst_sent = fst_sent.replace(sub_id, id_name[sub_id])  # replace other ids with actual name

                    # clean first sentence
                    fst_sent = clean_text(fst_sent)

                    # obtain the second sentence as a supplementary
                    scd_sent = sents[i + 1] if i < len(sents) - 1 else sents[i - 1]
                    sub_ids = prog.findall(scd_sent)
                    for sub_id in sub_ids:
                        scd_sent = scd_sent.replace(sub_id, id_name[sub_id])

                    # clean second sentence
                    scd_sent = clean_text(scd_sent)

                    # store each record into dict
                    # remove descriptions to keep record light, desc can be derived in name_desc dict by answer entity
                    record = {
                        's1': fst_sent,
                        's2': scd_sent,
                        'c_ans': candidates,
                        'ans': answer
                    }
                    data.append(record)
    return data


def main():
    entity_file = os.path.join(source_dir, 'entities.txt')
    corpus_file = os.path.join(source_dir, 'corpus.txt')
    id_name, name_desc = prepro_entities(entity_file, dump_id_name='id_name.json', dump_name_desc='name_desc.json')
    data = prepro_corpus(corpus_file, id_name)
    with open('all_data.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
