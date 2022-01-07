import os
import argparse
import pandas as pd
from corpora import *
from joblib import load
import evaluation_metrics


"""Retrieve results file"""
def retrieve_results_file(results_path):
	if os.path.isfile(results_path):
		results = pd.read_csv(results_path)
	else:
		results = pd.DataFrame(columns = ['cls', 'arch', 'ML', 'hl', 'roc_auc', 'pr_auc', 'params'])
	return results



"""Function for updating the results file with new results"""
def update_results_file(results_file, data, params):
	# Adding/updating result:
	if ((results_file['cls'] == data['cls']) & (results_file['arch'] == data['arch']) & (results_file['params'] == params) & (results_file['ML'] == data['ML'])).any():
		results_file.loc[(results_file['cls'] == data['cls']) & (results_file['arch'] == data['arch']) & (results_file['params'] == params) & (results_file['ML'] == data['ML'])] = [pd.Series(data)]
	else:
		results_file = results_file.append(data, ignore_index=True)

	return results_file



# """Model testing main function"""
def test_model(config):

    # Load data:
    corpus = SingleCorpus(config)

	# Retrieving results file:
    results_file = retrieve_results_file( 'Results.csv')
    dst_path = os.path.join('/home/user/data/Dataset/AudioClassification/Minz/shallow_cls', config.dataset, config.model_type)

    assert os.path.isfile(os.path.join(dst_path, config.shallow_model + '_{}.cls'.format(config.ml))), "No trained {}_{}.cls model for {} arch".format(config.shallow_model, config.ml, config.model_type)
    cls = load(os.path.join(dst_path, config.shallow_model + '_{}.cls'.format(config.ml)))


    # Prediction model:
    y_pred_prob = cls.predict_proba(corpus.x_test_NC[corpus.test_indexes])


    if type(y_pred_prob) == list:
        # Probabilities as given by the classifier to probability matrix (n_instances, n_classes):
        y_prob_matrix = np.array([[1-u[0] for u in single_class] for single_class in y_pred_prob]).T
    else:
        y_prob_matrix = y_pred_prob.toarray()

    # Eval:
    out_metrics = evaluation_metrics.compute_metrics(config = config, y_true = corpus.y_test_onehot[corpus.test_indexes], y_prob_matrix = y_prob_matrix)

    # Classifier parameters:
    params = str(cls.get_params())
    out_metrics['params'] = params
    out_metrics['ML'] = config.ml


    # Update results file:
    results_file = update_results_file(results_file, out_metrics, params)

    # Sorting results file:
    results_file = results_file.sort_values(['cls', 'arch', 'ML'], ascending = (True, True, True))

    # Write results file:
    results_file.to_csv('Results.csv', index = False)



    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'msd', 'jamendo'])
    parser.add_argument('--model_type', type=str, default='fcn',
                        choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn'])
    parser.add_argument('--shallow_model', type=str, default='forest',
                        choices=['forest', 'knn', 'SVM', 'xgboost'])
    parser.add_argument('--ml', type=str, default='powerset',
                        choices=['powerset', 'binary'])

    config = parser.parse_args()


    test_model(config)
