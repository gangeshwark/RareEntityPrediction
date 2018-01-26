from pycorenlp import StanfordCoreNLP


class CoreNLP(object):
    """Stanford Parser for information extraction and phrase detection
    Start Stanford Server:
        download stanford corenlp from: https://stanfordnlp.github.io/CoreNLP/download.html
        cd <Stanford CoreNLP folder>
        nohup java -Xmx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 > log 2>&1&
    """

    def __init__(self, properties=None):
        self.properties = properties if properties is not None else {
            'annotators': 'tokenize,ssplit,pos,lemma',
            'outputFormat': 'json'}
        self.parser = StanfordCoreNLP('http://localhost:9000')
        self.instance = None

    def annotate(self, text):
        self.instance = self.parser.annotate(text, properties=self.properties)

    def word_tokenize(self):
        words_list = []
        for sentence in self.instance['sentences']:
            words_list.append([token['word'] for token in sentence['tokens']])
        return words_list

    def sent_tokenize(self):
        return [' '.join(words) for words in self.word_tokenize()]

    def word_lemmatize(self):
        lemmas_list = []
        for sentence in self.instance['sentences']:
            lemmas_list.append([token['lemma'] for token in sentence['tokens']])
        return lemmas_list

    def sent_lemmatize(self):
        return [' '.join(lemmas) for lemmas in self.word_lemmatize()]
