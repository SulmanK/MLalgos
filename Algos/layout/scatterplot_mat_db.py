#----------------- Packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

#----------------- Function
""" Functions to create a confusion matrix from the predicted labels and then plot a bar plot"""


def scatterplot_matrix_db(X_train, y_train, clf, title):
    """Creates a scatter-plot matrix with decision boundaries """

    # Parameters
    n_classes = len(np.unique(y_train))
    plot_colors = ['purple', 'blue', 'green']
    plot_step = 0.5
    fig = plt.figure(figsize=(20, 10))

    # Iterate through all feature combinations
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                    [1, 2], [1, 3], [2, 3]]):

        # Set X and Y
        X = X_train[:, pair]
        y = y_train

        # Fit classifier
        clf.fit(X=X, y=y)

        # Plot the decision boundary
        fig_ax = fig.add_subplot(2, 3, pairidx + 1)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot contour of decision boundary
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = fig_ax.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn)

        # Set figure parameters
        fig_ax.set_xlabel('Feature ' + str(pair[0]), fontsize=14)
        fig_ax.set_ylabel('Feature ' + str(pair[1]), fontsize=14)
        plt.suptitle(title, fontsize=22)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=4.0)

        # Legend Patches
        purple_patch = mpatches.Patch(color='purple', label='Class 0')
        green_patch = mpatches.Patch(color='green', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 2')
        patches = [purple_patch, green_patch, blue_patch]

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            fig_ax.scatter(X[idx, 0], X[idx, 1], c=color, label=y[i],
                           cmap=plt.cm.PuBuGn, edgecolor='black', s=100)
            legend = fig_ax.legend(handles=patches, loc='upper right',
                                   borderaxespad=0., fontsize=14)

    return fig
