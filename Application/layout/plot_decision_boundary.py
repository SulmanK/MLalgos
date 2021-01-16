#------------------- Packages
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import tkinter


#------------------- Functions
def make_meshgrid(x, y, h=.02):
    # Creates a mesh grid used for plotting the contours in the decision boundary

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    # Plots the contours of the mesh grid for the decision boundary

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def scatter_plot_db(X, Y, y_train, X_test, num_neighbors, clf, y_pred_mine, y_pred_sklearn, accuracy_mine, accuracy_sklearn):
    # Function to create a side by side scatter plot showing decision boundaries for classification

    # Create a dictionary object to store each class with a color
    classes = np.unique(y_train)
    num_classes = len(np.unique(y_train))
    color_list = ['r', 'b', 'g', 'm', 'y', 'k', 'c']
    d = {a: b for a, b in zip(classes, color_list)}

    # Color each training sample with a class
    color_labels = []

    for i in y_train:
        for j, k in zip(d.keys(), d.values()):
            if i == j:
                color_labels.append(k)

    # Color map of the decision boundaries and labels
    if num_classes == 2:
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

        # build the legend
        red_patch = mpatches.Patch(color='red', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 2')

        # Legend labels
        patches = [red_patch, blue_patch]

    elif num_classes == 3:
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])

        # build the legend
        red_patch = mpatches.Patch(color='red', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 2')
        green_patch = mpatches.Patch(color='green', label='Class 3')

        # set up for handles declaration
        patches = [red_patch, blue_patch, green_patch]

    elif num_classes == 4:
        cmap_light = ListedColormap(
            ['#FFAAAA', '#AAAAFF', '#AAFFAA', '#ffccff'])
        cmap_bold = ListedColormap(
            ['#FF0000', '#0000FF', '#00FF00', '#FF00FF'])

        # build the legend
        red_patch = mpatches.Patch(color='red', label='Class 1')
        blue_patch = mpatches.Patch(color='blue', label='Class 2')
        green_patch = mpatches.Patch(color='green', label='Class 3')
        magenta_patch = mpatches.Patch(color='magenta', label='Class 4')

        # set up for handles declaration
        patches = [red_patch, blue_patch, green_patch, magenta_patch]

    # Plot figiure

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    # plt.subplots_adjust(left=0, bottom=0, right= -1, top=0, wspace=0.0, hspace=0.0)

    fig.suptitle("%i-Class KNN classification (k = %i)"
                 % (num_classes, num_neighbors), fontsize=22)

    # Plot my implementation and sci-kit learns model side-by-side

    # Iterate through the two axes and different y_preds from my model and sci-kit learns model to plot
    for ax, y_pred, title, clf_ in zip(axes.flatten(), [y_pred_mine, y_pred_sklearn],
                                       ['Mine (Accuracy = %i%%)' % (accuracy_mine),
                                        'Sklearn (Accuracy = %i%%)' % (accuracy_sklearn)],
                                       clf):

        # Assign color of testing set
        pred_color_labels = []
        for i in y_pred:
            for j, k in zip(d.keys(), d.values()):
                if i == j:
                    pred_color_labels.append(k)

        scat = ax.scatter(X, Y,  color=color_labels, marker='o',
                          linestyle='None', cmap=cmap_bold)
        fig.canvas.draw()

        # Plot testing set with predicted labels
        for i, j in zip(X_test, pred_color_labels):
            ax.scatter(i[0], i[1], c=j, marker='v', linewidth=3)

        # Plot Decision Boundary
        # Set-up grid for plotting.
        X0, X1 = X, Y
        xx, yy = make_meshgrid(X0, X1)

        # Plot decision boundaries
        plot_contours(ax, clf_, xx, yy, cmap=cmap_light, alpha=0.4)

        # Set axes limits
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        # Set titles
        ax.set_xlabel("Axes 1", fontsize=18)
        ax.set_ylabel("Axes 2", fontsize=18)
        ax.set_title(title, fontsize=18)

    # alternative declaration for placing legend outside of plot
        legend = ax.legend(handles=patches, bbox_to_anchor=(
            0.77, 1.00), loc=2, borderaxespad=0., fontsize=16)

    return fig
