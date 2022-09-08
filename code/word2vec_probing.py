from utils import load_all_samples, train_eval
import numpy as np
import argparse
import torch
import os

'''
This method returns the embedding of a word if it is in the vocabulary, 
otherwise returns an empty array of the word embeddings size.
'''

def get_word_embedding(word_embeddings, embedding_dim, word):
    try:
        word_vector = np.asarray(word_embeddings[word])
    except KeyError:
        word_vector = np.zeros(embedding_dim)
    return word_vector


'''
Computation of a sentence embedding by averaging the word embeddings
of its tokens. If none of the tokens are in the vocabulary, the sentence embedding
is an empty array of the word embeddings size.
'''

def get_sentence_embeddings(word_embeddings, embedding_dim, sentence):
    token_embeddings = []
    for token in sentence:
        try:
            word_vector = np.asarray(word_embeddings[token])
            token_embeddings.append(word_vector)
        except KeyError:
            pass    # Words out of the vocabulary are not considered for the creation of the sentence embedding

    if token_embeddings:
        sentence_embeddings = np.mean(token_embeddings, axis=0)
    else:
        sentence_embeddings = np.zeros(embedding_dim)
    
    return sentence_embeddings

'''
Creation of a dictionary for the word embeddings.
The word and word embeddings values must be splitted by tabs.
'''

def load_word_embeddings(src_path):
    word_embeddings = {}
    for line in open(src_path, 'r'):
        line = (line.strip()).split('\t')
        word = line[0]  # The first element of the line is the word
        embedding = [float(el) for el in line[1:]] # the other elements are the components of the word embedding, separated by tabs
        word_embeddings[word] = embedding
    
    return word_embeddings


'''
Extraction of events and sentence embeddings (if required) for each sample of the dataset.
'''
def extract_samples_representations(samples, word_embeddings, args):

    for sample in samples:
        sample.representation['source_embedding'] = get_word_embedding(word_embeddings, args.embedding_dim, sample.term_1)
        sample.representation['target_embedding'] = get_word_embedding(word_embeddings, args.embedding_dim, sample.term_2)
        if args.use_sentence_embeddings:
            sample.representation['sentence_embedding_1'] = get_sentence_embeddings(word_embeddings, args.embedding_dim, sample.sentence_1.tokens)
            sample.representation['sentence_embedding_2'] = get_sentence_embeddings(word_embeddings, args.embedding_dim, sample.sentence_2.tokens)




'''
Creation of the list of features and labels for training the SVM.
The list of features is created by the concatenation of events and sentence embeddings (if required).
'''
def create_features_vectors(samples, args):
    features = []
    labels = []
    for sample in samples:
        embeddings = sample.representation
        if args.use_sentence_embeddings:
            features.append(np.concatenate([embeddings['source_embedding'], embeddings['target_embedding'], embeddings['sentence_embedding_1'], embeddings['sentence_embedding_2']]))
        else:
            features.append(np.concatenate([embeddings['source_embedding'], embeddings['target_embedding']]))
        
        labels.append(sample.label)

    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_file')
    parser.add_argument('-ts', '--test_file')
    parser.add_argument('-o', '--output_file')
    parser.add_argument('-e', '--word_embeddings_path')
    parser.add_argument('-d', '--embedding_dim', type=int)
    parser.add_argument('-v', '--no_use_vague', action='store_true', default=False)
    parser.add_argument('-s', '--use_sentence_embeddings', action='store_true')

    args = parser.parse_args()

    print('\n\nUsing entence Embeddings:',args.use_sentence_embeddings)
    print('\nUsing class Vague:', not args.no_use_vague)
    print('\n\n')

    train_samples, labels_list = load_all_samples(args.train_file, 0, args)                         # Loading of train and test samples from
    valid_samples, test_label_list = load_all_samples(args.test_file, len(train_samples), args)     # train and test source files
    
    all_labels = list(set(test_label_list).union(set(labels_list)))

    print('Loading word embeddings...')
    word_embeddings = load_word_embeddings(args.word_embeddings_path)   # Creation of word embeddings dictionary

    print('Extracting sentences representations...')
    extract_samples_representations(train_samples, word_embeddings, args)   # Computation of samples representation using
    extract_samples_representations(valid_samples, word_embeddings, args)   # word embeddings
    
    all_labels += ['macro avg', 'weighted avg', 'accuracy']

    scores = {label: [] for label in all_labels}
        
    X_train, y_train = create_features_vectors(train_samples, args)     # Creation of SVM inputs
    X_test, y_test = create_features_vectors(valid_samples, args)
    classification_report = train_eval(X_train, y_train, X_test, y_test)    # SVM training and evaluation

    for label in all_labels:
        if label == 'accuracy':
            scores[label].append(str(classification_report[label]))
        else:
            if label in classification_report:
                scores[label].append(str(classification_report[label]['f1-score']))
            else:
                scores[label].append(str(None))
             
    with open(args.output_file, 'w+') as out_file:          # The output file will contain a row for each class f1-score and a row for 
        for label in scores:                                # macro avg, weighted avg and accuracy (in this case is equal to global f1-score)
            out_file.write('\t'.join([label]+scores[label])+'\n')

if __name__ == '__main__':
    main()