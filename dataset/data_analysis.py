from dataset.build_vocab_emb import load_json
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from collections import OrderedDict
import os
import re

home = os.path.expanduser('~')
source_dir = os.path.join(home, 'data', 'rare_entity')

prog = re.compile(r'(9202a8c04000641f8\w+)')


def count_length():
    c_s_dict = {}
    c_w_dict = {}
    with open(os.path.join(source_dir, 'corpus.txt'), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='process corpus'):
            sents = sent_tokenize(line)[0:2]  # split paragraph into sentences
            for sent in sents:
                words = word_tokenize(sent)
                if len(words) in c_s_dict:
                    c_s_dict[len(words)] += 1
                else:
                    c_s_dict[len(words)] = 1
                for word in words:
                    if len(word) in c_w_dict:
                        c_w_dict[len(word)] += 1
                    else:
                        c_w_dict[len(word)] = 1
    c_s_dict = OrderedDict(sorted(c_s_dict.items()))
    c_w_dict = OrderedDict(sorted(c_w_dict.items()))
    print('corpus sentences dict:')
    print(c_s_dict)
    print('corpus words dict:')
    print(c_w_dict)
    print('statistic information:')
    sum1, sum2, total_sum = 0, 0, 0
    for key, value in c_s_dict.items():
        if key <= 120:
            sum1 += value
        if key <= 100:
            sum2 += value
        total_sum += value
    print("len <= 120: {}, len <= 100: {}, total sents: {}, ratio 120: {:04.2f}, ratio 100: {:04.2f}"
          .format(sum1, sum2, total_sum, float(sum1) / float(total_sum) * 100, float(sum2) / float(total_sum) * 100))
    sum1, sum2, total_sum = 0, 0, 0
    for key, value in c_w_dict.items():
        if key <= 45:
            sum1 += value
        if key <= 40:
            sum2 += value
        total_sum += value
    print("len <= 45: {}, len <= 40: {}, total sents: {}, ratio 45: {:04.2f}, ratio 40: {:04.2f}"
          .format(sum1, sum2, total_sum, float(sum1) / float(total_sum) * 100, float(sum2) / float(total_sum) * 100))
    print('\n\n')

    d_s_dict = {}
    d_w_dict = {}
    with open(os.path.join(source_dir, 'entities.txt'), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='process entities'):
            fb_id, *_, name, desc = line.strip().split('\t')
            desc = sent_tokenize(desc, language='english')[0]  # NLTK sentence tokenization, get first sentence
            words = word_tokenize(desc)
            if len(words) in d_s_dict:
                d_s_dict[len(words)] += 1
            else:
                d_s_dict[len(words)] = 1
            for word in words:
                if len(word) in d_w_dict:
                    d_w_dict[len(word)] += 1
                else:
                    d_w_dict[len(word)] = 1
    d_s_dict = OrderedDict(sorted(d_s_dict.items()))
    d_w_dict = OrderedDict(sorted(d_w_dict.items()))
    print('entities sentences dict:')
    print(d_s_dict)
    print('entities words dict:')
    print(d_w_dict)
    print()
    sum1, sum2, total_sum = 0, 0, 0
    for key, value in d_s_dict.items():
        if key <= 60:
            sum1 += value
        if key <= 50:
            sum2 += value
        total_sum += value
    print("len <= 60: {}, len <= 50: {}, total sents: {}, ratio 60: {:04.2f}, ratio 50: {:04.2f}"
          .format(sum1, sum2, total_sum, float(sum1) / float(total_sum) * 100, float(sum2) / float(total_sum) * 100))
    sum1, sum2, total_sum = 0, 0, 0
    for key, value in d_w_dict.items():
        if key <= 35:
            sum1 += value
        if key <= 30:
            sum2 += value
        total_sum += value
    print("len <= 35: {}, len <= 30: {}, total sents: {}, ratio 35: {:04.2f}, ratio 30: {:04.2f}"
          .format(sum1, sum2, total_sum, float(sum1) / float(total_sum) * 100, float(sum2) / float(total_sum) * 100))


def count_candidates_num():
    cand_dict = {}
    with open(os.path.join(source_dir, 'corpus.txt'), 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='process corpus'):
            all_ids = prog.findall(line)
            nums = len(all_ids)
            if nums in cand_dict:
                cand_dict[nums] += 1
            else:
                cand_dict[nums] = 1
    cand_dict = OrderedDict(sorted(cand_dict.items(), key=lambda val: val[0], reverse=False))
    print(cand_dict)


def main():
    count_length()


if __name__ == '__main__':
    main()
