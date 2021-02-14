#----------------- Packages
import matplotlib.pyplot as plt
import numpy as np
#----------------- Function
""" Functions to create a confusion matrix from the predicted labels and then plot a bar plot"""


def confusion_matrix_dict(y_test, y_pred):
    """Creates a dictionary object of the confusion matrix from the predicted labels"""
    classes = np.unique(y_test)
    tmp_list = []
    for i, j in zip(y_test, y_pred):
        # True Positive
        if (i == classes[0]) & (j == classes[0]):
            tmp_list.append('TP')

        # True Negative
        elif (i == classes[1]) & (j == classes[1]):
            tmp_list.append('TN')

        # False Positive
        elif (i == classes[1]) & (j == classes[0]):
            tmp_list.append('FP')

        # False Negative
        elif (i == classes[0]) & (j == classes[1]):
            tmp_list.append('FN')

    # Create the dict object to zip across
    unique, counts = np.unique(tmp_list, return_counts=True)
    confusion_matrix = dict(zip(unique, counts))

    return confusion_matrix


def confusion_matrix_plot(y_test, y_pred_1, y_pred_2):
    "Side-by-side confusion matrix bar plot"
    # Set the values of the confusion matrices
    cm_mine = confusion_matrix_dict(y_test=y_test, y_pred=y_pred_1)
    cm_sklearn = confusion_matrix_dict(y_test=y_test, y_pred=y_pred_2)

    # Create a side-by-side confusion matrix bar plot between my implementation and sklearns
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    for ax, cm_, title_, in zip(axes.flatten(),
                                [cm_mine, cm_sklearn],
                                ['Confusion Matrix (Mine)', 'Confusion Matrix (Sklearn)']):

        # Create the location, label, and values of the bar plots
        loc = np.arange(4)
        labels = [x for x in cm_.keys()]
        values = [x for x in cm_.values()]

        # Set the colors
        colors = ['salmon', 'palegreen', 'red', 'green']

        # Create bar plot figure
        bar_plot = ax.bar(loc, values, tick_label=labels,
                          color=colors, log=True)

        # Set figure parameters
        ax.set_title(title_, fontsize=18)
        ax.set_ylabel('Frequency (Log)', fontsize=16)
        ax.set_xlabel('Label', fontsize=16)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Annotate each value of the bars
        for p in ax.patches:
            ax.annotate(np.round(p.get_height(), decimals=2),
                        (p.get_x()+p.get_width()/2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                        fontsize=16)

    return fig
