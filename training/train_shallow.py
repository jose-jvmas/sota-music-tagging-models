import multiprocessing
multiprocessing.set_start_method('spawn', True)

import os
import sklearn
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier



from joblib import dump
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from sklearn.metrics import average_precision_score

from scipy.sparse import issparse

def average_precision_wrapper(estimator, X, y):
    y_pred = estimator.predict(X)
    if issparse(y_pred): y_pred = y_pred.toarray()
    return average_precision_score(y, y_pred)

class SingleCorpus():

    def read_partition(self, config, partition):
        # Source path:
        read_path = os.path.join('embeddings', config.dataset, config.model_type)

        # Embeddings:
        with open(os.path.join(read_path, partition + '_emb.csv')) as fin:
            FileIn = fin.readlines()
        x = np.array([[float(u) for u in line.split(",")] for line in FileIn])


        # GT:
        with open(os.path.join(read_path, partition + '_gt.csv')) as fin:
            FileIn = fin.readlines()
        y = np.array([[float(u) for u in line.split(",")] for line in FileIn])

        return x, y

    def __init__(self, config):
        self.x_train_NC, self.y_train_onehot = self.read_partition(config, partition = 'train')
        self.train_indexes = list(range(len(self.x_train_NC)))
        
        self.x_valid_NC, self.y_valid_onehot = self.read_partition(config, partition = 'valid')
        self.valid_indexes = list(range(len(self.x_valid_NC)))
        
        self.x_test_NC, self.y_test_onehot = self.read_partition(config, partition = 'test')
        self.test_indexes = list(range(len(self.x_test_NC)))





"""Random Forest classifier"""
def forest_classifier(corpus, ml, dst_path):
    
    # Create model:
    forest = RandomForestClassifier(random_state=1)

    # Single class into multi-target:
    if ml == 'binary':
        multi_target_forest_model = MultiOutputClassifier(forest, n_jobs=-1)
        # Parameter(s) to optimize:
        params = {'estimator__n_estimators' : list(map(int, np.linspace(100, 1000, 10)))}
    elif ml == 'powerset':
        multi_target_forest_model = LabelPowerset(classifier = forest,
            require_dense=[False, False])
        # Parameter(s) to optimize:
        params = {'classifier__n_estimators' : list(map(int, np.linspace(100, 1000, 10)))}

    # Preparing data for parameter grid search:
    ### Merging train and validation partitions:
    X_GridSearch = np.vstack([corpus.x_train_NC[corpus.train_indexes], corpus.x_valid_NC[corpus.valid_indexes]])
    y_GridSearch = np.vstack([corpus.y_train_onehot[corpus.train_indexes], corpus.y_valid_onehot[corpus.valid_indexes]])

    ### Setting which is the validation data:
    valid_fold = [-1 for _ in range(corpus.x_train_NC[corpus.train_indexes].shape[0])] \
        + [0 for _ in range(corpus.x_valid_NC[corpus.valid_indexes].shape[0])]

    ### Setting PredefinedSplit:
    ps = PredefinedSplit(valid_fold)

    ### Score to optimize:
    score = 'average_precision'

    # Grid search
    #multi_target_forest = GridSearchCV(multi_target_forest_model, params, cv = ps, n_jobs = -1, scoring = score)
    multi_target_forest = GridSearchCV(multi_target_forest_model, params, cv = ps, n_jobs = -1, scoring = average_precision_wrapper)

    # Train model:
    multi_target_forest.fit(X_GridSearch, np.array(y_GridSearch, dtype=int))

    # Persisting the model:
    dump(multi_target_forest.best_estimator_, os.path.join(dst_path, 'forest_{}.cls'.format(ml)))

    return













def train(config):

    # Load data:
    corpus = SingleCorpus(config)
    dst_pth = os.path.join('shallow_cls', config.dataset, config.model_type)
    if not os.path.exists(dst_pth): os.makedirs(dst_pth)
    forest_classifier(corpus, 'powerset', dst_pth)

    # Train model:
    # RF = RandomForestClassifier()
    # RF.fit(x_train, y_train)

    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'msd', 'jamendo'])
    parser.add_argument('--model_type', type=str, default='fcn',
                        choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn'])
    parser.add_argument('--shallow_model', type=str, default='random_forest',
                        choices=['random_forest'])

    config = parser.parse_args()


    train(config)