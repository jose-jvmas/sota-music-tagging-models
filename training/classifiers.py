import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from joblib import dump
from skmultilearn.adapt import MLkNN
from sklearn.svm import SVC




from sklearn.metrics import average_precision_score

from scipy.sparse import issparse


def average_precision_wrapper(estimator, X, y):
    y_pred = estimator.predict(X)
    if issparse(y_pred): y_pred = y_pred.toarray()
    return average_precision_score(y, y_pred)



"""K-Nearest Neighbor classifier""" # EXPLORING KD-TREE
def kNN_classifier(corpus, ml, dst_path):

    # Create model:
    kNN = KNeighborsClassifier()
    # Single class into multi-target:
    if ml == 'binary':
        multi_target_kNN_model = MultiOutputClassifier(kNN, n_jobs=-1)
        # Parameter(s) to optimize:
        params = {'estimator__n_neighbors' : range(1, 13, 2)}
    elif ml == 'powerset':
        multi_target_kNN_model = LabelPowerset(classifier = kNN,
            require_dense=[False, False])
        # Parameter(s) to optimize:
        params = {'classifier__n_neighbors' : range(1, 13, 2)}

    # Preparing data for parameter grid search:
    ### Merging train and validation partitions:
    X_GridSearch = np.vstack([corpus.x_train_NC[corpus.train_indexes], corpus.x_valid_NC[corpus.valid_indexes]])
    y_GridSearch = np.vstack([corpus.y_train_onehot[corpus.train_indexes], corpus.y_valid_onehot[corpus.valid_indexes]])

    ### Setting which is the validation data:
    valid_fold = [-1 for _ in range(corpus.x_train_NC[corpus.train_indexes].shape[0])] \
        + [0 for _ in range(corpus.x_valid_NC[corpus.valid_indexes].shape[0])]

    ### Setting PredefinedSplit:
    ps = PredefinedSplit(valid_fold)

    # Grid search
    multi_target_kNN = GridSearchCV(multi_target_kNN_model, params, cv = ps, n_jobs = -1, scoring = average_precision_wrapper)

    # Train model:
    multi_target_kNN.fit(X_GridSearch, y_GridSearch)

    # Persisting the model:
    dump(multi_target_kNN.best_estimator_, os.path.join(dst_path, 'knn_{}.cls'.format(ml)))

    return



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


    # Grid search
    multi_target_forest = GridSearchCV(multi_target_forest_model, params, cv = ps, n_jobs = -1, scoring = average_precision_wrapper)

    # Train model:
    multi_target_forest.fit(X_GridSearch, np.array(y_GridSearch, dtype=int))

    # Persisting the model:
    dump(multi_target_forest.best_estimator_, os.path.join(dst_path, 'forest_{}.cls'.format(ml)))

    return





"""Support Vector Machine"""
def svm_classifier(corpus, ml, dst_path):
    
    # Create model:
    SVM = SVC(random_state=1, probability = True)

    # Single class into multi-target:
    if ml == 'binary':
        multi_target_SVM_model = MultiOutputClassifier(SVM, n_jobs=-1)
        # Parameter(s) to optimize:
        params = {'estimator__C' : list(np.linspace(0.5, 5, 10)),
                  'estimator__kernel' : ['rbf', 'linear', 'poly']}
    elif ml == 'powerset':
        multi_target_SVM_model = LabelPowerset(classifier = SVM,
            require_dense=[False, False])
        # Parameter(s) to optimize:
        params = {'classifier__C' : list(np.linspace(0.5, 5, 10)),
                  'classifier__kernel' : ['rbf', 'linear', 'poly']}

    # Preparing data for parameter grid search:
    ### Merging train and validation partitions:
    X_GridSearch = np.vstack([corpus.x_train_NC[corpus.train_indexes], corpus.x_valid_NC[corpus.valid_indexes]])
    y_GridSearch = np.vstack([corpus.y_train_onehot[corpus.train_indexes], corpus.y_valid_onehot[corpus.valid_indexes]])

    ### Setting which is the validation data:
    valid_fold = [-1 for _ in range(corpus.x_train_NC[corpus.train_indexes].shape[0])] \
        + [0 for _ in range(corpus.x_valid_NC[corpus.valid_indexes].shape[0])]

    ### Setting PredefinedSplit:
    ps = PredefinedSplit(valid_fold)


    # Grid search
    multi_target_SVM = GridSearchCV(multi_target_SVM_model, params, cv = ps, n_jobs = -1, scoring = average_precision_wrapper)

    # Train model:
    multi_target_SVM.fit(X_GridSearch, np.array(y_GridSearch, dtype=int))

    # Persisting the model:
    dump(multi_target_SVM.best_estimator_, os.path.join(dst_path, 'SVM_{}.cls'.format(ml)))

    return



if __name__ == '__main__':
    from corpora import SingleCorpus


    class confClass():
        def __init__(self):
            self.dataset = 'mtat'
            self.model_type = 'fcn'
            self.shallow_model = 'SVM'
            self.ml = 'binary'
    config = confClass() 
    
    # Load data:
    corpus = SingleCorpus(config)
    
    # Classifier:
    svm_classifier(corpus, config.ml, './')