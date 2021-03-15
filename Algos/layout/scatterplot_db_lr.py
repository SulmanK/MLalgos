#----------------- Packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sklearn.metrics import accuracy_score

#----------------- Function


def ScatterPlotDB_lr(X_train, y_train, X_test, y_test, clf_1, title_1):
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
        red_patch = mpatches.Patch(color='red', label='Negative Class 0')
        green_patch = mpatches.Patch(color='green', label='Positive Class 1')

        # Figure
        # Contour
        plot_step = 0.01
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))


        # Fit classifier
        clf.fit(X_train[:, 0:2], y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot contour
        ax.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn)

        # Find the indices for each class label
        y_pos = np.where(y_train == 1)
        y_neg = np.where(y_train == 0)

        # Plot the respective class labels
        ax.scatter(X_train[y_pos][:, 0], X_train[y_pos][:, 1],
                   marker='o', color='green',
                   edgecolor='black', s=500,
                   label='Positive +1')
        ax.scatter(X_train[y_neg][:, 0], X_train[y_neg][:, 1],
                   marker='o', color='red',
                   edgecolor='black', s=500,
                   label='Negative 0')

        # Test classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        full_title = title_ + ' (Accuracy = %i%%)' % (acc_score)

        # Set figure parameter
        ax.set_xlabel("$x_1$", fontsize=18)
        ax.set_ylabel("$x_1$", fontsize=18)
        ax.set_title(full_title, fontsize=18)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Assign legend patches to plot
        patches = [red_patch, green_patch]

        # Set legend
        legend = ax.legend(handles=patches, loc='upper right',
                           borderaxespad=0., fontsize=12)

    # Classes > 2
    elif classes > 2:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

        # Instantiate parameters for figure
        ax = ax
        clf = clf_1
        title_ = title_1

        # Legend Patches
        red_patch = mpatches.Patch(color='red', label='Class 0')
        green_patch = mpatches.Patch(color='green', label='Class 1')
        purple_patch =mpatches.Patch(color ='purple', label = 'Class 2')


        # Figure

        # Contour
        plot_step = 0.01
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Fit classifier
        clf.fit(X_train[:, 0:2], y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot contour of decision boundary
        ax.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn)

        # Find the indices for each class label
        y_2 = np.where(y_train == 2)
        y_1 = np.where(y_train == 1)
        y_0 = np.where(y_train == 0)

        # Plot the respective class labels
        ax.scatter(X_train[y_2][:, 0], X_train[y_2][:, 1],
                   marker='o', color='purple',
                   edgecolor='black', s=500,
                   label='Class 1', cmap = plt.cm.PuBuGn)

        ax.scatter(X_train[y_1][:, 0], X_train[y_1][:, 1],
                   marker='o', color='green',
                   edgecolor='black', s=500,
                   label='Class 1', cmap = plt.cm.PuBuGn)
        ax.scatter(X_train[y_0][:, 0], X_train[y_0][:, 1],
                   marker='o', color='r',
                   edgecolor='black', s=500,
                   label='Class 0', cmap = plt.cm.PuBuGn)



        # Test classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        full_title = title_ + ' (Accuracy = %i%%)' % (acc_score)

        # Set figure parameter
        ax.set_xlabel("$x_1$", fontsize=18)
        ax.set_ylabel("$x_1$", fontsize=18)
        ax.set_title(full_title, fontsize=18)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Assign legend patches to plot
        patches = [red_patch, green_patch, purple_patch]

        # Set legend
        legend = ax.legend(handles=patches, loc='upper right',
                           borderaxespad=0., fontsize=12)

    return fig
