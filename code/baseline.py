from utils import load_all_samples
from collections import Counter
import argparse
import random
import os


'''
Search for the most firequent class
'''
def find_most_common_class(train_samples, labels_list):
    labels_counter = Counter({label: 0 for label in labels_list})

    for sample in train_samples:
        labels_counter[sample.label] += 1
    
    return labels_counter.most_common(1)[0][0]
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_file')
    parser.add_argument('-ts', '--test_file')
    parser.add_argument('-o', '--output_file')
    parser.add_argument('-v', '--no_use_vague', action='store_true', default=False)
    
    args = parser.parse_args()

    train_samples, labels_list = load_all_samples(args.train_file, 0, args)             # Loading of train and test samples from
    valid_samples, _ = load_all_samples(args.test_file, len(train_samples), args)       # train and test source files


    most_common_class = find_most_common_class(train_samples, labels_list)      # Finds the most common class in the training set
    valid_labels = [sample.label for sample in valid_samples]                   # List of the true values 
    predicted_labels = [most_common_class for _ in range(len(valid_labels))]    # The predicted values are all equals to the most common class
    accuracy = sum(1 for x,y in zip(valid_labels, predicted_labels) if x == y) / float(len(valid_labels))
    
    with open(args.output_file, 'w+') as out_file:
        out_file.write('accuracy\t%f\n' % accuracy)

if __name__ == '__main__':
    main()
        
