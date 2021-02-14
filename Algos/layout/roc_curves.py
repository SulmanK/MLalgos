#------------------- Packages
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

#------------------- Functions
"""Calculates the FPR/TPR ratios to plot the ROC curves"""
def ROC_ratios(y_test, y_pred_prob):
    """Calculates the FPR / TPR for the ROC curves"""

    # Discretize y_test
    unique = np.unique(y_test)
    num_classes = len(np.unique(y_test))
    y_test_discrete = [0 if x == unique[0] else 1 for x in y_test]

    # Get threshold values from sci-kit learn to get a complete overview from 0 - 1.0
    false_positive_rate, true_positive_rate, threshold = roc_curve(
        y_test_discrete, y_pred_prob[:, 1])
    prob_scores = y_pred_prob[:, 1]
    classes = np.unique(y_test)
    TPR_rate = []
    FPR_rate = []

    # Iterate through each threshold, and calculate the FPR/TPR
    for i in threshold:
        tmp_label = []
        for j in prob_scores:
            # If the predicted probability is greater than the threshold, set it to class 2
            if j >= i:
                tmp = classes[1]
                tmp_label.append(tmp)

            # Else set it to class 1
            else:
                tmp = classes[0]
                tmp_label.append(tmp)
        # Next we go through the list of labels predicted by the thresholds, and assign FP, TP, FN, TN
        tmp_list = []
        for k, l in zip(y_test, tmp_label):
            # True Positive
            if (k == classes[1]) & (l == classes[1]):
                tmp_list.append('TP')

            # True Negative
            elif (k == classes[0]) & (l == classes[0]):
                tmp_list.append('TN')

            # False Posiive
            elif (k == classes[0]) & (l == classes[1]):
                tmp_list.append('FP')

            # False Negative
            elif (k == classes[1]) & (l == classes[0]):
                tmp_list.append('FN')

        # Create a dictionary of the counts for FP/TP/FN/TN
        unique, counts = np.unique(tmp_list, return_counts=True)
        y_pred_class = dict(zip(unique, counts))

        # Logic to assign a value of FP/TP/FN/TN to 0 if its not present
        if 'TP' not in y_pred_class:
            y_pred_class['TP'] = 0

        if 'FP' not in y_pred_class:
            y_pred_class['FP'] = 0

        if 'FN' not in y_pred_class:
            y_pred_class['FN'] = 0

        if 'TN' not in y_pred_class:
            y_pred_class['TN'] = 0

        # Calculate the TPR/FPR ratios
        TPR = (y_pred_class['TP']/(y_pred_class['TP'] + y_pred_class['FN']))
        FPR = (y_pred_class['FP']/(y_pred_class['TN'] + y_pred_class['FP']))

        # Append it to the final list for TPR / FPR ratios
        TPR_rate.append(TPR)
        FPR_rate.append(FPR)

    return FPR_rate, TPR_rate,

def roc_curves_side_by_side(y_test, pred_prob_1, pred_prob_2):
    """Plots the ROC curves of two implementations (mine and sklearns side by side"""

    # Get ROC ratios
    fpr_mine, tpr_mine = ROC_ratios(y_test, pred_prob_1)
    fpr_sklearn, tpr_sklearn = ROC_ratios(y_test, pred_prob_2)

    # Get ROC_AUC scores
    roc_auc_mine = roc_auc_score(y_test, pred_prob_1[:, 1])
    roc_auc_sklearn = roc_auc_score(y_test, pred_prob_2[:, 1])

    # Get ideal ROC coordinates
    ideal_roc = [x/10 for x in range(0, 11, 1)]

    # Get legend patches
    red_patch = mpatches.Patch(color='red', label='Ideal')
    blue_patch = mpatches.Patch(color='blue', label='Mine')
    green_patch = mpatches.Patch(color='green', label='Sklearn')
    pred_patches = [blue_patch, green_patch]

    # Create a side-by-side ROC curve plot between my implementation and sklearns
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    for ax, fpr, tpr, title_, color_, patches_ in zip(axes.flatten(),
                                                      [fpr_mine, fpr_sklearn], [
                                                          tpr_mine, tpr_sklearn],
                                                      ['ROC_AUC Score = {0:.3g}% (Mine)'.format(roc_auc_mine*100),
                                                       'ROC_AUC Score = {0:.3g}% (Sklearn)'.format(roc_auc_sklearn*100)],
                                                      ['blue', 'green'], pred_patches):
        # Create ROC plot
        scat = ax.plot(fpr, tpr,  color=color_, marker='x',
                       linestyle='solid')

        # Create ideal ROC plot
        standard = ax.plot(ideal_roc, ideal_roc,
                           color='red', linestyle = 'dotted')

        # Set figure parameter
        ax.set_xlabel("False Positive Ratio", fontsize=18)
        ax.set_ylabel("True Positive Ratio", fontsize=18)
        ax.set_title(title_, fontsize=18)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Assign legend patches to plot
        patches = [red_patch, patches_]

        # Set legend
        legend = ax.legend(handles=patches, loc='right',
                           borderaxespad=0., fontsize=16)

    return fig
