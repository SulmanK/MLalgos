#----------------- Packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sklearn.metrics import accuracy_score

#----------------- Function


def ScatterPlotDB_DT(X_train, y_train, X_test, y_test, clf_1, title_1):
    """
    Scatter plot with Decision boundary of a dataset

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
            The input data.

    y_train : array-like, shape (n_samples, n_features)
            The input data.

    clf_1 : class object
            Classifier of my LR model 

    title_1 : string
            Title of my LR model


    Returns
    -------
    Figure object.
    """

    # Instantiate Figure object
    classes = len(np.unique(y_train))


    # Two Classes
    if classes == 2:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

        # Instantiate parameters for figure
        ax = ax
        clf = clf_1
        title_ = title_1

        # Legend Patches
        red_patch = mpatches.Patch(color='red', label='Training Negative Class 0')
        green_patch = mpatches.Patch(color='green', label='Training Positive Class 1')
        lightcoral_patch = mpatches.Patch(color='lightcoral', label='Testing Negative Class 0')
        limegreen_patch = mpatches.Patch(color='limegreen', label='Testing Positive Class 1')


        # Figure


        # Plot Training Labels
        # Find respective class indices
        y_pos = np.where(y_train == ' <=50K')[0]
        y_neg = np.where(y_train == ' >50K')[0]

        # Plot the respective class labels
        ax.scatter(X_train.values[y_pos][:, 0], X_train.values[y_pos][:, 11],
                   marker='o', color='green',
                   edgecolor='black', s=500,
                   label='Training Positive +1')
        ax.scatter(X_train.values[y_neg][:, 0], X_train.values[y_neg][:, 11],
                   marker='o', color='red',
                   edgecolor='black', s=500,
                   label='Training Negative 0')

        # Test classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        full_title = title_ + ' (Accuracy = %i%%)' % (acc_score)

        # Plot testing labels
        # Find respective class labels
        y_pos = np.where(y_pred == ' <=50K')[0]
        y_neg = np.where(y_pred == ' >50K')[0]
        
        ax.scatter(X_test.values[y_pos][:, 0], X_test.values[y_pos][:, 11],
                   marker='o', color='limegreen',
                   edgecolor='black', s=200,
                   label='Testing Positive +1')
        ax.scatter(X_test.values[y_neg][:, 0], X_test.values[y_neg][:, 11],
                   marker='o', color='lightcoral',
                   edgecolor='black', s=200,
                   label='Testing Negative 0')

        # Set figure parameter
        ax.set_xlabel("$x_1$", fontsize=18)
        ax.set_ylabel("$x_1$", fontsize=18)
        ax.set_title(full_title, fontsize=18)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Assign legend patches to plot
        patches = [red_patch, green_patch, lightcoral_patch, limegreen_patch]

        # Set legend
        legend = ax.legend(handles=patches, loc='upper right',
                           borderaxespad=0., fontsize=12)


    return fig