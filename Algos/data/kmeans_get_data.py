# -------------------- Packages
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# -------------------- Function


def get_sample_data(num_classes, num_samples):
    """
    Loads the data depending on the amount of classes and number of samples set in the widgets.

    Parameters
    ----------
    num_classes : integer
        The number of classes in the data.

	num_samples : integer
        The number of samples in the data.

    Returns
    -------
    X_tr : array-like, shape (n_samples, n_features)
        The X training data..
    y_tr : array-like, shape (n_samples,)
        The training labels.
    X_te : array-like, shape (n_samples, n_features)
    	The X testing data.
    y_te : array-like, shape (n_samples,)
        The testing labels.

    """
    # Get data
    X, y = make_blobs(n_samples=num_samples, centers=num_classes,
     				  n_features=2, random_state=0, center_box=(0, 5))

    # Subset the data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                              test_size=0.2, random_state=42)
    return X_tr, X_te, y_tr, y_te
