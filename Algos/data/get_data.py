#-------------------- Packages
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#-------------------- Function
def get_sample_data(num_classes, num_samples):
    X, y = make_blobs(n_samples=num_samples, centers=num_classes, n_features=2,
                      random_state=0, center_box=(0, 5))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                              test_size=0.2, random_state=42)
    return X_tr, X_te, y_tr, y_te
