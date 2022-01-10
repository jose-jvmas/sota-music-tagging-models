import os
import itertools
import numpy as np
from sklearn import metrics

"""Performance measurement"""
def get_auc(est_array, gt_array):
    roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')

    return roc_aucs, pr_aucs


"""Load NN modality & data partition"""
def load_modality(corpus, model, partition):
    src_path = os.path.join('decision', corpus, model)

    with open(os.path.join(src_path, partition + '_est.csv')) as f:
        est = np.array([[float(u) for u in single_line.split(",")] for single_line in f.readlines()])

    with open(os.path.join(src_path, partition + '_gt.csv')) as f:
        gt = np.array([[int(u) for u in single_line.split(",")] for single_line in f.readlines()])


    return est, gt


"""Fusing N modalities using given weights"""
def fuse_n_modalities(modalities, weights):
    fused_modality = np.zeros(shape=modalities[0].shape,dtype=float)
    for it_mod in range(len(modalities)):
        fused_modality += modalities[it_mod] * weights[it_mod]

    return fused_modality


"""Function for iteratively updating weights (OPTIONS A and B)"""
def update_weights(weights):
    # Updating last weight:
    weights[-1] += 1

    # Checking all weights:
    it_weight = len(weights) - 1 
    while it_weight > 0:
        if weights[it_weight] <= 10:
            break
        else:
            weights[it_weight] = 0
            it_weight -= 1
            weights[it_weight] += 1 


    return weights


"""Fusing different modalities (validation partition) to retrieve the best weights (OPTION A)"""
def fuse_modalities(modalities, gt):
    # Number of modalities:
    n_mod = len(modalities)

    # Vector for the weights:
    current_weights = np.zeros(shape=(n_mod), dtype=int)

    best_metrics = (0, 0)
    best_weights = list()

    while current_weights[0] < 11:
        if sum(current_weights) == 10:  # Using current weights (if applicable):
            # Fusion:
            res_fusion = fuse_n_modalities(modalities, current_weights/10)
            
            # Evaluation:
            fusion_metrics = get_auc(res_fusion, gt)
            print("\t-Params {} -> ROC : {:.4f} | PR : {:.4f}".format(current_weights/10, fusion_metrics[0], fusion_metrics[1])) 
            if fusion_metrics[0] > best_metrics[0]:
                best_metrics = fusion_metrics
                best_weights = current_weights/10

        current_weights = update_weights(current_weights)
    
    print("\n**BEST RESULTS**\n\t-Weights: {}\n\t-Metrics : {}".format(best_weights, best_metrics))
    
    return best_weights, best_metrics


"""Individual weight per model"""
def experiment_single_weight_per_model(corpus):
    base_modalities = ['attention', 'crnn', 'fcn', 'hcnn', 'musicnn', 'sample', 'se', 'short']

    # Obtaining all possible combinations:
    cases = list()
    for i in range(2, len(base_modalities)):
        cases.extend(list(itertools.combinations(base_modalities, i)))

    with open('Results_SingleWeightPerModel.csv','w') as fout:
        fout.write("Combination,Weights,VAL_ROC_AUC,VAL_PR_AUC,TEST_ROC_AUC,TEST_PR_AUC\n")

    # Iterating through possible combinations:
    for single_case in cases:
        print("Current case {}".format(single_case))
        # Loading Validation GT:
        _, val_gt = load_modality(corpus, single_case[0], 'valid')
        
        # Loading validation data:
        modalities_val = list()
        for single_modality in single_case:
            est, _ = load_modality(corpus, single_modality, 'valid')
            modalities_val.append(est)

        # Obtaining optimal weights using validation partition:
        weights, val_fusion_metrics = fuse_modalities(modalities_val, val_gt)

        # Loading Test GT:
        _, test_gt = load_modality(corpus, single_case[0], 'test')
        
        # Loading validation data:
        modalities_test = list()
        for single_modality in single_case:
            est, _ = load_modality(corpus, single_modality, 'test')
            modalities_test.append(est)

        # Fusing modalities:
        res_fusion = fuse_n_modalities(modalities_test, weights)
        test_fusion_metrics = get_auc(res_fusion, test_gt)
        print("TEST -> ROC : {:.4f} | PR : {:.4f}".format(test_fusion_metrics[0], test_fusion_metrics[1])) 

        # Writing results:
        with open('Results_SingleWeightPerModel.csv','a') as fout:
            fout.write("{},{},{},{},{},{}\n".format(single_case, weights, val_fusion_metrics[0], val_fusion_metrics[1], test_fusion_metrics[0], test_fusion_metrics[1]))


    return



"""Individual weight per label and model"""
def experiment_single_weight_per_label_and_model(corpus):
    base_modalities = ['attention', 'crnn', 'fcn', 'hcnn', 'musicnn', 'sample', 'se', 'short']

    # Obtaining all possible combinations:
    cases = list()
    for i in range(2, len(base_modalities)):
        cases.extend(list(itertools.combinations(base_modalities, i)))

    with open('Results_SingleWeightPerModelAndLabel.csv','w') as fout:
        fout.write("Combination,VAL_ROC_AUC,VAL_PR_AUC,TEST_ROC_AUC,TEST_PR_AUC\n")

    # Iterating through possible combinations:
    for single_case in cases:
        print("Current case {}".format(single_case))

        # Loading Validation GT:
        _, val_gt = load_modality(corpus, single_case[0], 'valid')

        # Creating Validation vector of modalities:
        modalities_val = list()
        for single_modality in single_case:
            est, _ = load_modality(corpus, single_modality, 'valid')
            modalities_val.append(est)

        # Number of modalities and labels:
        n_mod = len(modalities_val)
        n_labels = modalities_val[0].shape[1]

        # Optimal labels initialization:
        optimal_weights = np.zeros(shape = (n_mod, n_labels))

        # Iterating through the different labels:
        for it_label in range(n_labels):
            print("\t-Processing label #{}".format(it_label))

            # Extracting the relevant labels of each modality:
            single_label_modalities = [modalities_val[it_mod][:, it_label] for it_mod in range(n_mod)]

            # Vector for the weights:
            current_weights = np.zeros(shape=(n_mod), dtype=int)

            best_metrics = (0, 0)
            best_weights = list()
            while current_weights[0] < 11:
                if sum(current_weights) == 10:  # Using current weights (if applicable):
                    # Fusion:
                    res_fusion = fuse_n_modalities(single_label_modalities, current_weights/10)
                    
                    # Evaluation:
                    fusion_metrics = get_auc(res_fusion, val_gt[:, it_label])
                    # print("\t-Params {} -> ROC : {:.4f} | PR : {:.4f}".format(current_weights/10, fusion_metrics[0], fusion_metrics[1])) 
                    if fusion_metrics[0] > best_metrics[0]:
                        best_metrics = fusion_metrics
                        best_weights = current_weights/10

                current_weights = update_weights(current_weights)

            # Saving best weights for current label:
            for it_mod in range(n_mod):
                optimal_weights[it_mod][it_label] = best_weights[it_mod]

        # Evaluating validation partition:
        fusion = np.zeros(shape=(modalities_val[0].shape), dtype = 'float')
        for it_mod in range(n_mod):
            fusion += modalities_val[it_mod]*optimal_weights[it_mod]
        val_fusion_metrics = get_auc(fusion, val_gt)


        # Loading Test GT:
        _, test_gt = load_modality(corpus, single_case[0], 'test')

        # Preparing test data:
        modalities_test = list()
        for single_modality in single_case:
            est, _ = load_modality(corpus, single_modality, 'test')
            modalities_test.append(est)

        # Fusing test modalities using weights from validation:
        fusion = np.zeros(shape=(modalities_test[0].shape), dtype = 'float')
        for it_mod in range(n_mod):
            fusion += modalities_test[it_mod]*optimal_weights[it_mod]

        # Evaluating performance:
        test_fusion_metrics = get_auc(fusion, test_gt)

         # Writing results:
        with open('Results_SingleWeightPerModelAndLabel.csv','a') as fout:
            fout.write("{},{},{},{},{}\n".format(single_case, val_fusion_metrics[0], val_fusion_metrics[1], test_fusion_metrics[0], test_fusion_metrics[1]))


    return


if __name__ == '__main__':
    corpus = 'mtat'
    experiment_single_weight_per_model(corpus) # Option A)

    # experiment_single_weight_per_label_and_model(corpus) # Option B)

    print("End")