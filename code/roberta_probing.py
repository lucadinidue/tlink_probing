from utils import load_all_samples, train_eval
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import argparse
import torch
import os


'''
Given all the sentence embeddings, computes the mean of sub-tokens embeddingst from stard_idx to end_idx.
Is used to compute event and sentence representations. 
'''
def compute_embedding_mean(all_embeddings, start_idx, end_idx):
    if start_idx == end_idx:
        return all_embeddings[start_idx]
    else:
        selected_embeddings = all_embeddings[start_idx:end_idx+1]
        return torch.mean(selected_embeddings, 0)


'''
Extraction of events and sentence embeddings (if required) for each sample of the dataset.
'''       

def extract_samples_representations(model_name, samples, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)#.to('cuda:0')
    num_hidden_layers = model.config.num_hidden_layers

    for sample in samples:
        if sample.id % 100 == 0:
            print(sample.id)
        encoded_tokens_1 = tokenizer.encode_plus(sample.sentence_1.text, return_tensors='pt')#.to('cuda:0') # The sentence is tokenized in sub-word
        encoded_tokens_2 = tokenizer.encode_plus(sample.sentence_2.text, return_tensors='pt')#.to('cuda:0') # to be given in input to the model
        
        source_token_ids = np.where(np.array(encoded_tokens_1.word_ids()) == sample.source_event_idx)[0]    # Each original token can be divided in
        target_token_ids = np.where(np.array(encoded_tokens_2.word_ids()) == sample.target_event_idx)[0]    # multiple sub-tokens by the previous
                                                                                                            # tokenization process. We get the 
                                                                                                            # sub-tokens corresponding to the original
                                                                                                            # token of each event.
        with torch.no_grad():                                                                                       
            model_output_1 = model(**encoded_tokens_1, output_hidden_states=True)
            model_output_2 = model(**encoded_tokens_2, output_hidden_states=True)
            hidden_states_1 = model_output_1.hidden_states  # There is an array for each model layer, the firts array
            hidden_states_2 = model_output_2.hidden_states  # is the input embedding layer
            for layer in range(1, num_hidden_layers+1):     # For each of the other layers we extract sample representations
                layer_output_1 = torch.squeeze(hidden_states_1[layer])
                layer_output_2 = torch.squeeze(hidden_states_2[layer])

                # To compute event representations, we compute the mean of all event's sub-tokens embeddings
                source_embedding = compute_embedding_mean(layer_output_1, source_token_ids[0], source_token_ids[-1]).cpu().detach().numpy()
                target_embedding = compute_embedding_mean(layer_output_2, target_token_ids[0], target_token_ids[-1]).cpu().detach().numpy()
                # To compute sentence representation, we compute the mean of all sentence subtokens except the special tokens
                # <s> and </s> placed at the beginning and end of the sentence by the tokenization process
                sentence_embedding_1 = compute_embedding_mean(layer_output_1, 1, -2).cpu().detach().numpy()
                sentence_embedding_2 = compute_embedding_mean(layer_output_2, 1, -2).cpu().detach().numpy()

                # Each sample will have a different representation extracted from each layer
                if args.use_sentence_embeddings:
                    sample.representation[layer] = {'source_embedding': source_embedding, 'target_embedding': target_embedding, 'sentence_embedding_1': sentence_embedding_1, 'sentence_embedding_2': sentence_embedding_2}
                else:
                    sample.representation[layer] = {'source_embedding': source_embedding, 'target_embedding': target_embedding} 


'''
Creation of the list of features and labels for training the SVM.
The list of features is created by the concatenation of events and sentence embeddings (if required).
'''

def create_features_vectors(samples, layer, args):
    features = []
    labels = []
    for sample in samples:
        embeddings = sample.representation[layer]
        if args.use_sentence_embeddings:
            features.append(np.concatenate([embeddings['source_embedding'], embeddings['target_embedding'], embeddings['sentence_embedding_1'], embeddings['sentence_embedding_2']]))
        else:
            features.append(np.concatenate([embeddings['source_embedding'], embeddings['target_embedding']]))
        labels.append(sample.label)

    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name',
                      help='the name of the model used for extracting representations.')
    parser.add_argument('-tr', '--train_file')
    parser.add_argument('-ts', '--test_file')
    parser.add_argument('-o', '--output_file')
    parser.add_argument('-f', '--first_layer', type=int, required=False)
    parser.add_argument('-l', '--last_layer', type=int, required=False)
    parser.add_argument('-v', '--no_use_vague', action='store_true', default=False)
    parser.add_argument('-s', '--use_sentence_embeddings', action='store_true')
    
    args = parser.parse_args()

    print('\n\nUsing entence Embeddings:',args.use_sentence_embeddings)
    print('\nUsing class Vague:', not args.no_use_vague)
    print('\n\n')

    model_string = args.model_name.split('/')[-1]

    train_samples, labels_list = load_all_samples(args.train_file, 0, args)                         # Loading of train and test samples from
    valid_samples, test_label_list = load_all_samples(args.test_file, len(train_samples), args)     # train and test source files
    all_labels = list(set(test_label_list).union(set(labels_list)))

    extract_samples_representations(args.model_name, train_samples, args)    # Computation of samples representation using
    extract_samples_representations(args.model_name, valid_samples, args)    # roBERTa contextual embeddings

    all_labels += ['macro avg', 'weighted avg', 'accuracy']
    
    model_config = AutoConfig.from_pretrained(args.model_name)    
    num_hidden_layers = model_config.num_hidden_layers 
    layers_scores = {label:[] for label in all_labels}

    for layer in range(1, num_hidden_layers+1):                 # We execute the probing extracting representation from each of roBERTa layers
        print('\n\n------ layer %d ------\n' % layer)
        X_train, y_train = create_features_vectors(train_samples, layer, args)      # Creation of SVM inputs
        X_test, y_test = create_features_vectors(valid_samples, layer, args)
        classification_report = train_eval(X_train, y_train, X_test, y_test)        # SVM training and evaluation

        for label in all_labels:
            if label == 'accuracy':
                layers_scores[label].append(str(classification_report[label]))
            else:
                if label in classification_report:
                    layers_scores[label].append(str(classification_report[label]['f1-score']))
                else:
                    layers_scores[label].append(str(None))
    
    with open(args.output_file, 'w+') as out_file:                  # The output file will contain a row for each class f1-score and a row for 
        for label in layers_scores:                                 # macro avg, weighted avg and accuracy (in this case is equal to global f1-score)
            out_file.write('\t'.join([label]+layers_scores[label])+'\n')    # each row will contain a value for each layer

if __name__ == '__main__':
    main()
