import os
import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score





    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

    # https://www.kaggle.com/kmkarakaya/multi-label-model-evaluation

    # # Minz metrics
    # roc_aucs  = roc_auc_score(y_true, y_pred_matrix, average = 'macro')
    # pr_aucs = average_precision_score(y_true, y_pred_matrix, average = 'macro')


    # np.average([average_precision_score(y_true[it,:], y_pred_matrix[it,:]) for it in range(y_true.shape[0])])
    # np.average([pr_score(y_true[it, :], y_pred_matrix[it, :]) for it in range(y_true.shape[0])])

    # np.average([auc_score(y_true[:, it], y_pred_matrix[:, it]) for it in range(y_true.shape[1])])
    # np.average([pr_score(y_true[:, it], y_pred_matrix[:, it]) for it in range(y_true.shape[1])])


"""Hamming Loss metric"""
def hamming_loss_eval(y_true, y_pred):
    er = hamming_loss(y_true = y_true, y_pred = y_pred)
    return er


"""Classification accuracy with area under ROC"""
def custom_roc_auc_score(y_true, y_pred, average = 'macro'):
    results = list()
    for it_class in range(y_true.shape[1]):
        if len(np.unique(y_true[:, it_class])) == 1:
            results.append(0.5)
        else:
            results.append(roc_auc_score(y_true[:, it_class], y_pred[:, it_class]))
    return np.average(results)


"""Area under ROC using the Precision/Recall curve"""
def custom_average_precision_score(y_true, y_pred, average = 'macro'):
    results = list()
    for it_class in range(y_true.shape[1]):
        if len(np.unique(y_true[:, it_class])) == 1:
            results.append(0.5)
        else:
            results.append(average_precision_score(y_true[:, it_class], y_pred[:, it_class]))
    return np.average(results)






"""Function for computing the different metrics"""
def compute_metrics(config, y_true, y_prob_matrix):
    
    data = {'cls' : config.shallow_model,
            'arch' : config.model_type
    }


    # Hamming loss metric
    # data['hl'] = hamming_loss_eval(y_true = y_true, y_pred = y_pred) 
    data['hl'] = '-'
    
    # ROC-AUC:
    data['roc_auc'] = custom_roc_auc_score(y_true = y_true, y_pred = y_prob_matrix, average = 'macro')

    # PR-AUC:
    data['pr_auc'] = custom_average_precision_score(y_true = y_true, y_pred = y_prob_matrix, average = 'macro')

    return data


if __name__ == '__main__':

    y_true = [
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 1, 0, 1]
    ]


    y_pred = [
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 1]
    ]


    print("Evaluation protocol")