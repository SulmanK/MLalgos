#----------------- Packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

#----------------- Function


def ScatterPlotDB_svm(X_train, y_train, clf_1, clf_2, title_1, title_2):
    """
    Scatter plot with Decision boundary of a dataset

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
            The input data.

    y_train : array-like, shape (n_samples, n_features)
            The input data.

    clf_1 : class object
            Classifier of my SVM model 

    clf_2 : class object
            Classifier of sklearn SVM model

    title_1 : string
            Title of my SVM model

    title_2 : string
            Title of sklearn SVM model

    Returns
    -------
    Figure object.
    """

    # Instantiate Figure object
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    # Create list of clf_lists / title_lists
    clf_list = [clf_1, clf_2]
    title_list = [title_1, title_2]

    # Legend Patches
    red_patch = mpatches.Patch(color='red', label='Negative Class -1')
    green_patch = mpatches.Patch(color='green', label='Positive Class +1')
    blue_patch = mpatches.Patch(
        color='blue', label='SVM Decision Boundary (Mine)')
    purple_patch = mpatches.Patch(
        color='purple', label='SVM Decision Boundary (Sklearn)')
    pred_patches = [blue_patch, purple_patch]

    # Iterate through through the subplot axes to populate our scatterplot
    for ax, clf, title_, color_, patches_ in zip(axes.flatten(),
                                                 clf_list,
                                                 title_list,
                                                 ['blue', 'Purple'],
                                                 pred_patches):

        plot_step = 0.01
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot contour of decision boundary
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn)

        # Find the indices for each class label
        y_pos = np.where(y_train == 1)
        y_neg = np.where(y_train == -1)

        # Plot the respective class labels
        ax.scatter(X_train[y_pos][:, 0], X_train[y_pos][:, 1],
                   marker='o', color='green',
                   edgecolor='black', s=500,
                   label='Positive +1')
        ax.scatter(X_train[y_neg][:, 0], X_train[y_neg][:, 1],
                   marker='o', color='r',
                   edgecolor='black', s=500,
                   label='Negative -1')

        # Set figure parameter
        ax.set_xlabel("$x_1$", fontsize=18)
        ax.set_ylabel("$x_1$", fontsize=18)
        ax.set_title(title_, fontsize=18)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Assign legend patches to plot
        patches = [red_patch, green_patch, patches_]

        # Set legend
        legend = ax.legend(handles=patches, loc='upper right',
                           borderaxespad=0., fontsize=12)

    return fig
