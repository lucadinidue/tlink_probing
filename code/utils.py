from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
import torch
import re

class Sentence():

    def __init__(self, raw_text):
        self.text = raw_text
        self.raw_text = self.detokenize_sentence(raw_text)
        self.tokens = self.get_tokens(raw_text)

    def detokenize_sentence(self, sent):
        sent = re.sub(r'\s([?.!,\'](?:\s|$))', r'\1', sent)
        sent = re.sub(r'\ \'', r"'", sent)
        sent = re.sub(r'\sn\'t', r"n't", sent)
        return sent   

    def get_tokens(self, raw_text):
        return raw_text.split(' ')

class Sample():

    def __init__(self, sample_id, line):
        self.id = sample_id
        self.sentence_1 = Sentence(line[0])
        self.sentence_2 = Sentence(line[1])
        self.source_event_idx = int(line[2][3:])
        self.target_event_idx = int(line[4][3:])
        self.term_1 = line[3]
        self.term_2 = line[5]
        self.label = line[6]
        self.representation = dict()
    
    def __str__(self):
        return 'Sent_1: %s\nSent_2: %s\nIdx_1: %d\nIdx_2: %d\nTerm_1: %s\nTerm_2: %s\nLabel: %s' % (self.sentence_1, self.sentence_2, self.source_event_idx, self.target_event_idx, self.term_1, self.term_2, self.label)

'''
Create a Sample object from each line of the input file.
Also returns the list of labels.
'''
def load_all_samples(src_path, id_start, args):
    samples = []
    i = id_start
    for line in open(src_path, 'r'):
        line = line.strip().split('\t')
        sample = Sample(i, line)
        if not args.no_use_vague or sample.label != 'VAGUE': # if we ar not using vague, all samples whit vague class are ignored
            samples.append(sample)
            i += 1
    label_list = set(sample.label for sample in samples)
    return samples, list(label_list)

'''
SVM training and evaluation.
'''
def train_eval(X_train, y_train, X_test, y_test):

    scaler = MinMaxScaler()
    
    scaled_X_train = scaler.fit_transform(X_train)  # Scaling must be applied both to training and evaluation set
    scaled_X_test = scaler.transform(X_test)        # using the same scale

    clf = LinearSVC(max_iter=50000, dual=False)
    clf.fit(scaled_X_train, y_train)
    
    y_pred = clf.predict(scaled_X_test)

    return classification_report(y_test, y_pred, output_dict=True)

