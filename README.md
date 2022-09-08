# Probing Temporal Relations Between Pairs of Events.

This is the public repository for the paper [How About Time? Probing a Multilingual Language Model for Temporal Relations](https://github.com/irenedini/tlink_probing/blob/main/COLING_tlink_probing.pdf) (COLING 2022).

The probing experiments have been run using a multilingual language model (XLM-RoBERTa). All experiments focused on temporal relation classifiction between pairs of events (either in the same sentence or across different sentence) for different languages: English, Italian, Spanish, and French. 


## Datasets

- English TimeBank (EN-TimeBank): We have used the TempEval-3 revised version of the corpus. The dataset is not more available online (the entire SemEval 2013 web page has been dismissed and all data have disapperared), so we make available the portions we have used (TimeBank corpus for training and TE3 Platinum test set).

- Spanish TimeBank (ES-TimeBank): The corpus (v1.0) and the data splits (train and test) can be obatined from [LDC](https://catalog.ldc.upenn.edu/LDC2012T12). Train and test data can be obrtained by runing the preprocesing scripts.

- English TimeBank Dense (EN-TB-Dense): The original dataset is available [here](). For our experiments we have re-used the EN-TB-Dense data from the [CATENA repository](https://github.com/paramitamirza/CATENA/blob/master/data/TimebankDense.TLINK.txt). We make available our train and test data.

- Italian TimeBank (IT-TimeBank): The Italian TimeBank (train and test split) can be obtained via [this form](https://hltdistributor.fbk.eu/index.php) from FBK. Train and test data can be obrtained by runing the preprocesing scripts.

- French TimeBank (FR-TimeBank): The original dataset was originally available via [this download link](https://gforge.inria.fr/projects/fr-timebank/). We invite those who wants to obtain the full corpus to [contact the authors](https://hal.inria.fr/inria-00606631/fr/). For replicability purpose, we make available the train and test data we have created.

The folder /data contains only datasets that we make available for replicability purposes and/or that is no more available online.

## Preprocessing

For each corpus, we have created dedicated scripts to extra the relevant data. They are available in the folder /preprocess 

## Probing

All probing experiments have been run using a linear SVM. The code is available in the folder /code

