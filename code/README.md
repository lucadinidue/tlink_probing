This folder contains 3 script:

1) baseline.py 
This script is used to compute a simple baseline: we always predict the most frequent class in the training set.

How to use:
python baseline.py --train_file TRAIN_PATH --test_file TEST_PATH --output_file OUTPUT_PATH

Creates a file in OUTPUT_PATH containing the accuracy of the baseline

2) word2vec_probing.py
This script computes the probing using word2vec representations (actually works with all static word embeddings)

How to use:
python word2vec_probing.py --train_file TRAIN_PATH --test_file TEST_PATH --word_embeddings_path EMBEDDINGS_PATH --embedding_dim 100 --output_file OUTPUT_PATH

EMBEDDINGS_PATH  points to the file containing word embeddings in the form "word\tword_embedding" and where the embedding components are splitted by tabs. The --embedding_dim parameter must be changed according to the embedding size.
The output file will contain a row for each class plus a row for macro avg, weighted avg and accuracy each containing the f1-score of the probing.

3) roberta_probing.py
This script computer the probing using roBERTa models (it works for every huggingface LM but you may have to change the creation of sentence representation because of different special tokens).

python roberta_probing.py --train_file TRAIN_PATH --test_file TEST_PATH --model_name xlm-roberta-base  --output_file OUTPUT_PAHT 

The output file will contain a row for each class plus a row for macro avg, weighted avg and accuracy. 
Each row will contain the f1-score of the probing for each layer, for a total of 12 values.

Script 2 and script 3 have 2 additional parameters:
--use_sentence_embeddings: used to use not only event embeddings but also sentence embeddings when creating samples representations.
--no_use_vague: used to ignore all samples with class VAGUE.
