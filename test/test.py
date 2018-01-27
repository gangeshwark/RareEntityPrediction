import re
from pprint import pprint
from nltk import word_tokenize, sent_tokenize, regexp_tokenize, wordpunct_tokenize

prog = re.compile(r'(9202a8c04000641f8\w+)')
id_replace = '**BLANK**'


def prepro_dict(corpus_file):
    data = set()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_ids = prog.findall(line)
            for fb_id in all_ids:
                data.add(fb_id)
    id_name = {}
    for i, d in enumerate(data):
        id_name[d] = 'SAMPLE_NAME_' + str(i)
    return id_name


def prepro_corpus(corpus_file, id_name):
    data = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_ids = prog.findall(line)
            if len(all_ids) > 10:  # filter out doc with more than 10 blanks
                continue
            candidates = [id_name[fb_id] for fb_id in all_ids]
            line = line.replace("``", '"').replace("''", '"')

            # one method (solve the problem, which can make sure the sentence is truly split, but some drawbacks caused)
            # line = ' '.join(wordpunct_tokenize(line))
            # sents = [sent for sent in sent_tokenize(line) if len(sent) >= 10]

            # alternative method (need to define what kind of regexp is better)
            line = ' '.join(regexp_tokenize(line, pattern="\w*?-?\w+-\w+|\w*'\w+|[a-zA-z]+\S+?|\S+"))
            sents = [' '.join(word_tokenize(sent)) for sent in sent_tokenize(line) if len(sent) >= 10]

            for i in range(len(sents)):
                if not prog.search(sents[i]):  # no freebase ids are contained
                    continue
                ids = prog.findall(sents[i])  # obtain all freebase ids in a sentence
                for fb_id in ids:
                    answer = id_name[fb_id]
                    other_ids = [x for x in ids if x != fb_id]  # get other freebase ids
                    fst_sent = sents[i].replace(fb_id, id_replace)  # replace target id with **BLANK**
                    for sub_id in other_ids:
                        fst_sent = fst_sent.replace(sub_id, id_name[sub_id])  # replace other ids with actual name
                    # obtain the second sentence as a supplementary
                    scd_sent = sents[i + 1] if i < len(sents) - 1 else sents[i - 1]
                    sub_ids = prog.findall(scd_sent)
                    for sub_id in sub_ids:
                        scd_sent = scd_sent.replace(sub_id, id_name[sub_id])
                    # store each record into dict
                    record = {
                        's1': fst_sent,
                        's2': scd_sent,
                        'c_ans': candidates,
                        'ans': answer
                    }
                    data.append(record)
    return data


def main():
    corpus_file = 'corpus_sample.txt'
    id_name = prepro_dict(corpus_file)
    data = prepro_corpus(corpus_file, id_name)
    for d in data:
        pprint(d)
        print('\n')


if __name__ == '__main__':
    main()
