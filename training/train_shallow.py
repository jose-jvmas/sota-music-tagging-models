# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

import os
import argparse
import numpy as np


from classifiers import *
from corpora import *





def train(config):

    # Load data:
    corpus = SingleCorpus(config)
    
    # Destination path:
    dst_pth = os.path.join('shallow_cls', config.dataset, config.model_type)
    if not os.path.exists(dst_pth): os.makedirs(dst_pth)

    # Training models:
    if config.shallow_model == 'forest':
        forest_classifier(corpus, config.ml, dst_pth)
    elif config.shallow_model == 'knn':
        kNN_classifier(corpus, config.ml, dst_pth)
    elif config.shallow_model == 'SVM':
        svm_classifier(corpus, config.ml, dst_pth)


    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'msd', 'jamendo'])
    parser.add_argument('--model_type', type=str, default='fcn',
                        choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn'])
    parser.add_argument('--shallow_model', type=str, default='forest',
                        choices=['forest', 'knn', 'SVM'])
    parser.add_argument('--ml', type=str, default='powerset',
                        choices=['powerset', 'binary'])

    config = parser.parse_args()


    train(config)