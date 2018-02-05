import json
from pprint import pprint

import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

"""
1. Clean the data (DONE)
(We will do the below 2 tasks separately just before processing the data for training.
2. Tokenize all sequences in the data and assign idx to the words. (TODO)
3. Pad sequences to a fixed number (TODO)
"""
special_character = re.compile(r'[^a-z_\d ]', re.IGNORECASE)

stops = set(stopwords.words("english"))
print("stops", len(stops))
stops = {k: 1 for k in stops}


def clean_text(text):
    text = word_tokenize(text.lower())
    # remove stop words.
    # text = [w for w in text if not w in stops]
    text1 = []
    for w in text:
        try:
            if stops[w]:
                pass
        except:
            text1.append(w)
    text = " ".join(text1)
    # remove Special Characters
    text = special_character.sub('', text)
    # replace multiple spaces with single one.
    text = re.sub(' +', ' ', text)
    # return the cleaned text
    return text


with open('../dataset/all_data_unclean.json', 'r') as f:
    data = json.load(f)
    print(type(data))
    print(len(data))
    pprint(data[0]['s1'])
    pprint(clean_text(data[0]['s1']))
    for x in tqdm(data):
        clean_text(x['s1'])
        clean_text(x['s2'])
        # clean_text(x[])
