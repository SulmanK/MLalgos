
#----------------- Packages

from model.dtClf import *
from model.dtReg import *
import numpy as np
import pandas as pd
import random


# ------------------------------ Helper Functions
def entropy(subset, total_count):
    """
    Calculates the Entropy given a subset of the data.

    Parameters
    ----------
    subset : Array
        Data.

    total_count : integer
        The total count of all data.

    Returns
    -------
    entropy - Integer
        The measure of entropy of the data.


    """

    # Count the number of samples in the subset
    count = int(len(subset))

    # Ratio of the count and total count

    ratio = float(count / total_count)

    # Entropy
    entropy = -ratio * np.log(ratio)

    return entropy


def gini(subset, total_count):
    """
    Calculates the Gini given a subset of the data.

    Parameters
    ----------
    subset : Array
        Data.

    total_count : integer
        The total count of all data.

    Returns
    -------
    gini - Integer
        The measure of gini impurity of the data.


    """

    # Count the number of samples in the subset
    count = int(len(subset))

    # Probability of success
    ratio_p = float(count / total_count)

    # Probability of failure
    ratio_q = 1 - ratio_p

    # Gini
    gini = ratio_p**2 + ratio_q**2

    # Gini Impurity
    gini_impurity = 1 - gini

    return gini_impurity


def label_selection(labels):
    """
    Assigns the label due to majority voting.

    Parameters
    ----------
    labels : Array
        Data.

    Returns
    -------
    majority_label - str
        Returns the majority label


    """

    # Unique / Frequency of labels
    unique, frequency = np.unique(labels,
                                  return_counts=True)

    # If there is only one kind of label, assign it to that label
    if len(unique) == 1:
        majority_label = unique

    # Else
    else:
        # Count of Label 1 is greater than Count of Label 2
        if frequency[0] > frequency[1]:

            # Assign Label 1
            majority_label = unique[0]

        # Count of Label 2 is greater than Count of Label 1
        elif frequency[1] > frequency[0]:
            majority_label = unique[1]

        else:
            # Randomly generated integer from 0 to 1
            j = random.randint(0, 1)

            # Assign the label using that randomly generated integer
            majority_label = unique[j]

    return majority_label


def suboptimal_classes_dict(classes, min_class_idx):
    """
    Dictionary to store the data for each suboptimal split

    Parameters
    ----------
    classes: List
        List of classes for the feature.

    min_class_idx: integer
        Index of the class with the minimum decision splitting method value.

    Returns
    -------    
    tmp_dict - dict
        Dictionary containing the data for each suboptimal split class

    """
    # Create a list of suboptimal class vlaues
    list_of_classes = [x for x in classes]
    tmp_suboptimal_classes = []
    for x in list_of_classes:
        if x != classes[min_class_idx]:
            tmp_suboptimal_classes.append(x)

    tmp_dict = {k: [] for k in tmp_suboptimal_classes}

    return tmp_dict


def decision_tree_split_classification(feature, label, method):
    """
    Splitting criteria of the decision tree.

    Parameters
    ----------
    feature: X array
        X Data.

    label: y array
        Y labels

    method: string 
        Weighting method for deciding split.

    Returns
    -------    
    weighted_method - integer
        The sum of all weighted values.

    optimal_split_data - array
        Subset of the data for the optimal split.

    suboptimal_split_data - array
        Subset of the data for the suboptimal splits.

    optimal_label - array
        Subset of the labels for the optimal split.

    suboptimal_label - array
        Subset of the labels for the suboptimal splits.

    """
    # Check if feature is string or categorical
    if (feature.dtype.name == 'str160') | (feature.dtype.name == 'object'):

        # Number of classes
        classes = np.unique(feature)

        # Number of samples
        total_count = int(len(feature))

        # List to store the splitting method values
        tmp_list = []

        # Store the total splitting method values across all the classes
        weighted_method = 0

        # Iterate through each class
        for i in classes:

            # Find the indices with correspond to each class
            idx = np.where(feature == i)

            # Subset the data by using the previous indices
            subset = feature.values[idx]

            # Splitting Method
            if method == 'Entropy':

                # Entropy of the ith class
                entropy_i = entropy(subset, total_count)

                # Append to list to find the minimum
                tmp_list.append(entropy_i)

                # Weighted average across all classes
                weighted_method += ((entropy_i * len(subset))/total_count)

            # Splitting Method
            elif method == 'Gini':

                # Gini measure of the ith class
                gini_i = gini(subset, total_count)

                # Append to list to find the minimum
                tmp_list.append(gini_i)

                # Weighted average across all classes
                weighted_method += ((gini_i * len(subset))/total_count)

        # Find the index corresponding to the minimum decision splitting method value
        tmp_array = tmp_list
        min_idx = np.argmin(tmp_array)

        # Optimal
        # Optimal split indices
        optimal_idx = np.where(feature == classes[min_idx])

        # Optimal split data
        optimal_split_data = optimal_idx

        # Optimal labels
        optimal_split_label = label.values[optimal_idx]

        # Majority label
        optimal_label = label_selection(optimal_split_label)

        # Create a list of the Optimal Class and Optimal Label
        optimal_label = [classes[min_idx], optimal_label]

        # Suboptimal
        # Function to create suboptimal_split_dict
        tmp_dict = suboptimal_classes_dict(classes, min_idx)

        # Iterate through each class of the dictionary
        for k in tmp_dict.keys():

            # Select the feature values which are the k class
            idx = np.where(feature == k)

            # Append those feature values to the dictionary
            tmp_dict[k].append(idx)

        # Assign suboptimal_split_data to the tmp dict
        suboptimal_split_data = tmp_dict

        # Suboptimal split labels
        tmp_dict = suboptimal_classes_dict(classes, min_idx)

        # Iterate through each class of the dictionary
        for k in tmp_dict.keys():

            # Select the feature values which are the k class
            idx = np.where(feature == k)

            # Labels for each suboptimal class
            suboptimal_split_label = label.values[idx]

            # Assign majority label
            suboptimal_label = label_selection(suboptimal_split_label)

            # Append values
            tmp_dict[k].append(suboptimal_label)

        # Assign suboptimal_label to the tmp dict
        suboptimal_label = tmp_dict

    # elif feature is continuous or integer dataype
    elif (feature.dtype.name != 'str160') & (feature.dtype.name != 'object'):
        # Number of samples
        total_count = len(feature)

        # Midpoint of the feature
        mid_point = np.mean(feature.values)

        # Subset of data < midpoint
        subset_before = feature.values[np.where(feature.values < mid_point)]

        # Subset of data > midpoint
        subset_after = feature.values[np.where(feature.values > mid_point)]

        # List to store the splitting method values
        tmp_list = []

        # Store the total splitting method values across all the classes
        weighted_method = 0

        # List of data containg subset < midpoint and subset > midpoint
        subset_list = [subset_before, subset_after]

        # Iterate through the two subsets
        for i in subset_list:

            # Method
            if method == 'Entropy':

                # Entropy of the ith class
                entropy_i = entropy(i, total_count)

                # Append to list to find the minimum
                tmp_list.append(entropy_i)

                # Weighted average across all classes
                weighted_method += ((entropy_i * len(i))/total_count)

            # Method
            elif method == 'Gini':

                # Gini of the ith class
                gini_i = gini(i, total_count)

                # Append to list to find the minimum
                tmp_list.append(gini_i)

                # Weighted average across all classes
                weighted_method += ((gini_i * len(i))/total_count)

        # Find the index corresponding to the minimum decision splitting method value
        tmp_array = tmp_list
        min_idx = np.argmin(tmp_array)

        # Subset of data points < midpoint
        if min_idx == 0:

            # Optimal
            # Optimal split indices
            optimal_idx = np.where(feature < mid_point)

            # Optimal split data
            optimal_split_data = optimal_idx

            # Optimal split labels
            optimal_split_label = label.values[optimal_idx]

            # Majority label
            optimal_label = label_selection(optimal_split_label)

            # List of Optimal Label < Midpoint
            optimal_label = ['<', mid_point, optimal_label]

            # Suboptimal
            # Indices of the suboptimal splits
            suboptimal_idx = np.where(feature > mid_point)

            # Suboptimal split data
            suboptimal_split_data = suboptimal_idx

            # Suboptimal split labels
            suboptimal_split_label = label.values[suboptimal_idx]

            # Majority label
            suboptimal_label = label_selection(suboptimal_split_label)

            # List of Suboptimal Label > Midpoint
            suboptimal_label = ['>', mid_point, suboptimal_label]

        # Subset of data points > midpoint
        elif min_idx == 1:

            # Optimal
            # Optimal split indices
            optimal_idx = np.where(feature > mid_point)

            # Optimal split data
            optimal_split_data = optimal_idx

            # Optimal split labels
            optimal_split_label = label.values[optimal_idx]

            # Majority label
            optimal_label = label_selection(optimal_split_label)

            # List of Optimal Label > Midpoint
            optimal_label = ['>', mid_point, optimal_label]

            # Suboptimal
            # Indices of the suboptimal splits
            suboptimal_idx = np.where(feature < mid_point)

            # Suboptimal split data
            suboptimal_split_data = suboptimal_idx

            # Suboptimal split labels
            suboptimal_split_label = label.values[suboptimal_idx]

            # Majority label
            suboptimal_label = label_selection(suboptimal_split_label)

            # List of Suboptimal Label < Midpoint
            suboptimal_label = ['<', mid_point, suboptimal_label]

    return weighted_method, optimal_split_data, suboptimal_split_data, optimal_label, suboptimal_label


def decision_tree_generate_classification(X_train, y_train, features, method):
    """
    Iterates through each feature of a decision tree to find the optimal tree layout.

    Parameters
    ----------
    X_train: X array
        X Data.

    y_train: y array
        Y labels

    features: array
        Array of features in the data.

    method: string 
        Weighting method for deciding split.

    Returns
    -------    
    feature - str
        The feature which has the minimum splitting criteria

    optimal_data - array
        Subset of the data for the optimal split.

    suboptimal_data - array
        Subset of the data for the suboptimal splits.

    optimal_label - array
        Subset of the labels for the optimal split.

    suboptimal_label - array
        Subset of the labels for the suboptimal splits.

    """
    # List of variables used for generating the decision tree layout
    min_weighted_method = []
    min_optimal_split_data = []
    min_suboptimal_split_data = []
    min_feature = []
    min_optimal_split_label = []
    min_suboptimal_split_label = []

    # Iterate through each feature to select the optimal/suboptimal data/labels
    for i in features:

        # Function of decision tree splitting
        weighted_method_i, optimal_split_data_i, suboptimal_split_data_i, optimal_label_i, suboptimal_label_i = decision_tree_split_classification(
            X_train[i], y_train, method)

        # Append parameters
        min_feature.append(i)
        min_weighted_method.append(weighted_method_i)
        min_optimal_split_data.append(optimal_split_data_i)
        min_suboptimal_split_data.append(suboptimal_split_data_i)
        min_optimal_split_label.append(optimal_label_i)
        min_suboptimal_split_label.append(suboptimal_label_i)

    # Specific case when there is zero or one feature left in iteration
    if (len(min_weighted_method) == 1) | (len(min_weighted_method) == 0):
        # Assign index to zero
        idx = 0

        # Feature
        feature = min_feature

        # Optimal data
        optimal_data = min_optimal_split_data

        # Optimal label
        optimal_label = min_optimal_split_label

        # Suboptimal data
        suboptimal_data = min_suboptimal_split_data

        # Suboptimal label
        suboptimal_label = min_suboptimal_split_label

    elif len(min_weighted_method) != 1:
        # Assign the index to the minimum decision splitting criteria
        idx = np.argmin(min_weighted_method)

        # Feature
        feature = min_feature[idx]

        # Optimal data
        optimal_data = min_optimal_split_data[idx]

        # Optimal label
        optimal_label = min_optimal_split_label[idx]

        # Suboptimal data
        suboptimal_data = min_suboptimal_split_data[idx]

        # Suboptimal label
        suboptimal_label = min_suboptimal_split_label[idx]

    return feature, optimal_data, suboptimal_data, optimal_label, suboptimal_label


def decision_tree_layout_classification(X_train, y_train, method):
    """
    The decision tree layout contains the roots/nodes of the features, and how the data is split into optimal / suboptimal

    Parameters
    ----------
    X_train: X array
        X Data.

    y_train: y array
        Y labels

    method: string 
        Weighting method for deciding split.

    Returns
    -------    
    feature_layout - list
        List of features that are the roots of the decision tree.

    optimal_data_dict - dict
        For each feature(root) contains the data labels.

    suboptimal_data_dict - dict
        For each feature(root) contains the data labels.


    """
    # List of the feature layout
    features_layout = []

    # List of features
    features = [x for x in X_train.columns]

    # Copy of feature list
    tmp = features

    # Assign X_train to data
    data = X_train

    # Assign y_train to labels
    labels = y_train

    # Create empty dictionaries of optimal/suboptimal
    optimal_data_dict = {k: [] for k in features}
    suboptimal_data_dict = {k: [] for k in features}

    # Continue iteration unless features (roots) < total_features and total number of samples is not equal to 1
    while (len(features_layout) <= len(tmp)) & (len(data) != 1):

        # Function of decision tree layout
        feature, optimal_data, suboptimal_data, optimal_label, suboptimal_label = decision_tree_generate_classification(
            data, labels, features, method)

        # Append feature to tree layout
        features_layout.append(feature)

        # Features that are not in decision tree layout
        features = []
        for k in tmp:
            if k not in features_layout:
                features.append(k)

        # Append labels to each feature
        optimal_data_dict[feature].append(optimal_label)
        suboptimal_data_dict[feature].append(suboptimal_label)

        # Assign data / labels
        data = data.iloc[optimal_data]
        labels = labels.iloc[optimal_data]

    return features_layout, optimal_data_dict, suboptimal_data_dict


def decision_tree_predict_classification(X_train, y_train, X_test, layout, optimal_data, suboptimal_data):
    """
    Predict function of the decision tree

    Parameters
    ----------
    X_train: X array
        X Data.

    y_train: y array
        Y labels

    X_test: X array
        X Data.

    layout: list
        Tree layout features(roots) 

    optimal_data - dict
        For each feature(root) contains the data labels.

    suboptimal_data - dict
        For each feature(root) contains the data labels.


    Returns
    -------    
    labels - array
        Assigned labels of the testing set.


    """

    # List of labels
    labels = []

    # Number of samples in the testing set
    m = len(X_test)

    # Iterate through each sample
    for i in range(0, m):

        # Index the testing sample
        x = X_test.iloc[i, :]

        # Initialize a counter to make sure that once the testing label is assigned a label, it will continue to next sample
        counter = 0
        while counter < 1:
            # Temporary variable
            num_pass = 0

            # Iterate through each feature
            for feature in layout:

                # If num_pass is equal to 1, we continue to next feature
                if num_pass == 1:
                    continue

                else:

                    # If the feature is a str or object
                    if (X_test[feature].dtype.name == 'str160') | (X_test[feature].dtype.name == 'object'):

                        # Select the data labels corresponding to that feature
                        tmp_opt = optimal_data[feature]

                        # If the testing sample follows the optimal data split continue to next feature
                        if (x[feature] == tmp_opt[0][0]) & (feature != layout[-1]):
                            continue

                        # Elif the testing sample follows the suboptimal data split assign it the majority label from dict
                        elif (x[feature] == tmp_opt[0][0]) & (feature == layout[-1]):
                            labels.append(tmp_opt[0][0])
                            num_pass = 1

                        # Edge case when there was no training data in the layout to assign a label
                        else:
                            # Try
                            try:
                                tmp_sub = suboptimal_data[feature][0][x[feature]][0]
                                labels.append(tmp_sub)
                                num_pass = 1

                            # When there is a keyerror
                            except KeyError:
                                # Since there is no data for that feature, take the majority class votes
                                idx = X_train[X_train[feature]
                                              == x[feature]].index.values
                                tmp = y_train[idx].value_counts().index[0]
                                labels.append(tmp)
                                num_pass = 1

                    # Elif the feature is integer
                    elif (X_test[feature].dtype.name != 'str160') & (X_test[feature].dtype.name != 'object'):

                        # Select the data labels corresponding to that feature
                        tmp_opt = optimal_data[feature]

                        # If the optimum data dictionary contains a greater than symbol
                        if tmp_opt[0][0] == '<':

                            # If the testing sample follows the optimal data split continue to next feature
                            if (x[feature] < tmp_opt[0][1]) & (feature != layout[-1]):
                                continue

                            # Elif the testing sample follows the optimal data split and is the last feature in the layout, assign the majority label
                            elif (x[feature] < tmp_opt[0][1]) & (feature == layout[-1]):
                                labels.append(tmp_opt[0][1])
                                num_pass = 1

                            # Another case is when the testing sample follows the suboptimal data split, append the majority label
                            else:
                                tmp_sub = suboptimal_data[feature]
                                labels.append(tmp_sub[0][2])
                                num_pass = 1

                        # Elif the optimum data dictionary contains a less than symbol
                        elif tmp_opt[0][0] == '>':

                            # If the testing sample follows the optimal data split continue to next feature
                            if (x[feature] > tmp_opt[0][1]) & (feature != layout[-1]):
                                continue

                            # Elif the testing sample follows the optimal data split and is the last feature in the layout, assign the majority label
                            elif (x[feature] > tmp_opt[0][1]) & (feature == layout[-1]):
                                labels.append(tmp_opt[0][1])
                                num_pass = 1

                            # Another case is when the testing sample follows the suboptimal data split, append the majority label
                            else:
                                tmp_sub = suboptimal_data[feature]
                                labels.append(tmp_sub[0][2])
                                num_pass = 1

                    # Assign counter to num_pass
                    counter = num_pass

    return np.array(labels, dtype=object)





def mse(subset):
    """
    Calculates the mse given a dataset

    Parameters
    ----------
    subset : Array
        Data.

    total_count : integer
        The total count of all data.

    Returns
    -------
    mse - Integer
        The measure of mse of the data.


    """
    # Count the number of samples in the subset
    count = int(len(subset))

    # Ratio of the count and total count
    avg_subset = np.mean(subset)

    # Mean Squared Error
    mse = np.mean((subset - avg_subset)**2)

    return mse


def decision_tree_split_regression(feature, label, method):
    """
    Splitting criteria of the decision tree.

    Parameters
    ----------
    feature: X array
        X Data.

    label: y array
        Y labels

    method: string 
        Weighting method for deciding split.

    Returns
    -------    
    weighted_method - integer
        The sum of all weighted values.

    optimal_split_data - array
        Subset of the data for the optimal split.

    suboptimal_split_data - array
        Subset of the data for the suboptimal splits.

    optimal_label - array
        Subset of the labels for the optimal split.

    suboptimal_label - array
        Subset of the labels for the suboptimal splits.

    """
    # Number of samples
    total_count = len(feature)

    # Midpoint of the feature
    mid_point = np.mean(feature.values)

    # Subset of data < midpoint
    subset_before = label.values[np.where(feature.values < mid_point)]

    # Subset of data > midpoint
    subset_after = label.values[np.where(feature.values > mid_point)]

    # List to store the splitting method values
    tmp_list = []

    # Store the total splitting method values across all the classes
    weighted_method = 0

    # List of data containg subset < midpoint and subset > midpoint
    subset_list = [subset_before, subset_after]

    # Iterate through the two subsets
    for i in subset_list:
        # Method
        if method == 'MSE':
            # Entropy of the ith class
            mse_i = mse(i)

            # Append to list to find the minimum
            tmp_list.append(mse_i)

            # Weighted average across all classes
            weighted_method += ((mse_i * len(i)))

    # Find the index corresponding to the minimum decision splitting method value
    tmp_array = tmp_list
    min_idx = np.argmin(tmp_array)

    # Subset of data points < midpoint
    if min_idx == 0:

        # Optimal
        # Optimal split indices
        optimal_idx = np.where(feature < mid_point)

        # Optimal split data
        optimal_split_data = optimal_idx

        # Optimal split labels
        optimal_split_label = label.values[optimal_idx]

        # Majority label
        optimal_label = np.mean(optimal_split_label)

        # List of Optimal Label < Midpoint
        optimal_label = ['<', mid_point, optimal_label]

        # Suboptimal
        # Indices of the suboptimal splits
        suboptimal_idx = np.where(feature > mid_point)

        # Suboptimal split data
        suboptimal_split_data = suboptimal_idx

        # Suboptimal split labels
        suboptimal_split_label = label.values[suboptimal_idx]

        # Majority label
        suboptimal_label = np.mean(suboptimal_split_label)

        # List of Suboptimal Label > Midpoint
        suboptimal_label = ['>', mid_point, suboptimal_label]

    # Subset of data points > midpoint
    elif min_idx == 1:

        # Optimal
        # Optimal split indices
        optimal_idx = np.where(feature > mid_point)

        # Optimal split data
        optimal_split_data = optimal_idx

        # Optimal split labels
        optimal_split_label = label.values[optimal_idx]

        # Majority label
        optimal_label = np.mean(optimal_split_label)

        # List of Optimal Label > Midpoint
        optimal_label = ['>', mid_point, optimal_label]

        # Suboptimal
        # Indices of the suboptimal splits
        suboptimal_idx = np.where(feature < mid_point)

        # Suboptimal split data
        suboptimal_split_data = suboptimal_idx

        # Suboptimal split labels
        suboptimal_split_label = label.values[suboptimal_idx]

        # Majority label
        suboptimal_label = np.mean(suboptimal_split_label)

        # List of Suboptimal Label < Midpoint
        suboptimal_label = ['<', mid_point, suboptimal_label]

    return weighted_method, optimal_split_data, suboptimal_split_data, optimal_label, suboptimal_label


def decision_tree_generate_regression(X_train, y_train, features, method):
    """
    Iterates through each feature of a decision tree to find the optimal tree layout.

    Parameters
    ----------
    X_train: X array
        X Data.

    y_train: y array
        Y labels

    features: array
        Array of features in the data.

    method: string 
        Weighting method for deciding split.

    Returns
    -------    
    feature - str
        The feature which has the minimum splitting criteria

    optimal_data - array
        Subset of the data for the optimal split.

    suboptimal_data - array
        Subset of the data for the suboptimal splits.

    optimal_label - array
        Subset of the labels for the optimal split.

    suboptimal_label - array
        Subset of the labels for the suboptimal splits.

    """
    # List of variables used for generating the decision tree layout
    min_weighted_method = []
    min_optimal_split_data = []
    min_suboptimal_split_data = []
    min_feature = []
    min_optimal_split_label = []
    min_suboptimal_split_label = []

    # Iterate through each feature to select the optimal/suboptimal data/labels
    for i in features:

        # Function of decision tree splitting
        weighted_method_i, optimal_split_data_i, suboptimal_split_data_i, optimal_label_i, suboptimal_label_i = decision_tree_split_regression(
            X_train[i], y_train, method)

        # Append parameters
        min_feature.append(i)
        min_weighted_method.append(weighted_method_i)
        min_optimal_split_data.append(optimal_split_data_i)
        min_suboptimal_split_data.append(suboptimal_split_data_i)
        min_optimal_split_label.append(optimal_label_i)
        min_suboptimal_split_label.append(suboptimal_label_i)

    # Specific case when there is zero or one feature left in iteration
    if (len(min_weighted_method) == 1) | (len(min_weighted_method) == 0):
        # Assign index to zero
        idx = 0

        # Feature
        feature = min_feature[0]

        # Optimal data
        optimal_data = min_optimal_split_data[idx]

        # Optimal label
        optimal_label = min_optimal_split_label[idx]

        # Suboptimal data
        suboptimal_data = min_suboptimal_split_data[idx]

        # Suboptimal label
        suboptimal_label = min_suboptimal_split_label[idx]

    elif len(min_weighted_method) != 1:
        # Assign the index to the minimum decision splitting criteria
        idx = np.argmin(min_weighted_method)

        # Feature
        feature = min_feature[idx]

        # Optimal data
        optimal_data = min_optimal_split_data[idx]

        # Optimal label
        optimal_label = min_optimal_split_label[idx]

        # Suboptimal data
        suboptimal_data = min_suboptimal_split_data[idx]

        # Suboptimal label
        suboptimal_label = min_suboptimal_split_label[idx]

    return feature, optimal_data, suboptimal_data, optimal_label, suboptimal_label


def decision_tree_layout_regression(X_train, y_train, method):
    """
    The decision tree layout contains the roots/nodes of the features, and how the data is split into optimal / suboptimal

    Parameters
    ----------
    X_train: X array
        X Data.

    y_train: y array
        Y labels

    method: string 
        Weighting method for deciding split.

    Returns
    -------    
    feature_layout - list
        List of features that are the roots of the decision tree.

    optimal_data_dict - dict
        For each feature(root) contains the data labels.

    suboptimal_data_dict - dict
        For each feature(root) contains the data labels.


    """
    # List of the feature layout
    features_layout = []

    # List of features
    features = [x for x in X_train.columns]

    # Copy of feature list
    tmp = features

    # Assign X_train to data
    data = X_train

    # Assign y_train to labels
    labels = y_train

    # Create empty dictionaries of optimal/suboptimal
    optimal_data_dict = {k: [] for k in features}
    suboptimal_data_dict = {k: [] for k in features}

    # Continue iteration unless features (roots) < total_features and total number of samples is not equal to 1
    while (len(features_layout) < len(tmp)) & (len(data) != 1):

        # Function of decision tree layout
        feature, optimal_data, suboptimal_data, optimal_label, suboptimal_label = decision_tree_generate_regression(
            data, labels, features, method)

        # Append feature to tree layout
        features_layout.append(feature)

        # Features that are not in decision tree layout
        features = []
        for k in tmp:
            if k not in features_layout:
                features.append(k)

        # Append labels to each feature
        optimal_data_dict[feature].append(optimal_label)
        suboptimal_data_dict[feature].append(suboptimal_label)

        # Assign data / labels
        data = data.iloc[optimal_data]
        labels = labels.iloc[optimal_data]

    return features_layout, optimal_data_dict, suboptimal_data_dict


def decision_tree_predict_regression(X_train, y_train, X_test, layout, optimal_data, suboptimal_data):
    """
    Predict function of the decision tree

    Parameters
    ----------
    X_train: X array
        X Data.

    y_train: y array
        Y labels

    X_test: X array
        X Data.

    layout: list
        Tree layout features(roots) 

    optimal_data - dict
        For each feature(root) contains the data labels.

    suboptimal_data - dict
        For each feature(root) contains the data labels.


    Returns
    -------    
    labels - array
        Assigned labels of the testing set.


    """

    # List of labels
    labels = []

    # Number of samples in the testing set
    m = len(X_test)

    # Iterate through each sample
    for i in range(0, m):

        # Index the testing sample
        x = X_test.iloc[i, :]

        # Initialize a counter to make sure that once the testing label is assigned a label, it will continue to next sample
        counter = 0
        while counter < 1:
            # Temporary variable
            num_pass = 0

            # Iterate through each feature
            for feature in layout:

                # If num_pass is equal to 1, we continue to next feature
                if num_pass == 1:
                    continue

                else:

                    # if the feature is integer
                    if (X_test[feature].dtype.name != 'str160') & (X_test[feature].dtype.name != 'object'):

                        # Select the data labels corresponding to that feature
                        tmp_opt = optimal_data[feature]

                        # If the optimum data dictionary contains a greater than symbol
                        if tmp_opt[0][0] == '<':

                            # If the testing sample follows the optimal data split continue to next feature
                            if (x[feature] < tmp_opt[0][1]) & (feature != layout[-1]):
                                continue

                            # Elif the testing sample follows the optimal data split and is the last feature in the layout, assign the majority label
                            elif (x[feature] < tmp_opt[0][1]) & (feature == layout[-1]):
                                labels.append(tmp_opt[0][1])
                                num_pass = 1

                            # Another case is when the testing sample follows the suboptimal data split, append the majority label
                            else:
                                tmp_sub = suboptimal_data[feature]
                                labels.append(tmp_sub[0][2])
                                num_pass = 1

                        # Elif the optimum data dictionary contains a less than symbol
                        elif tmp_opt[0][0] == '>':

                            # If the testing sample follows the optimal data split continue to next feature
                            if (x[feature] > tmp_opt[0][1]) & (feature != layout[-1]):
                                continue

                            # Elif the testing sample follows the optimal data split and is the last feature in the layout, assign the majority label
                            elif (x[feature] > tmp_opt[0][1]) & (feature == layout[-1]):
                                labels.append(tmp_opt[0][1])
                                num_pass = 1

                            # Another case is when the testing sample follows the suboptimal data split, append the majority label
                            else:
                                tmp_sub = suboptimal_data[feature]
                                labels.append(tmp_sub[0][2])
                                num_pass = 1

                    # Assign counter to num_pass
                    counter = num_pass

    return np.array(labels, dtype=object)









# ---------------- Class

class myDecisionTree:
    """
    Decision Tree.

    Parameters
    ----------

    method: string
        Specifies the supervised learning method solving - Classification or Regression problem.
        'Classification', 'Regression'

    splitting_method: string
        Specifies the decision splitting method to find the optimal decision tree structure.
        Classification = ['Entropy', 'Gini']
        Regression = ['MSE']
`


    Attributes
    ----------
    fit : model
        Training model.

    predict : model
        Use the training model to make predictions
    """

    def __init__(self, method, splitting_method):
        self.method = method
        self.splitting_method = splitting_method

    def fit(self, X, y):
        """
        Fit the training data (X,y).

        Parameters
        ----------
        X : dataframe, shape (n_samples, n_features)
            The input data.
        y : dataframe, shape (n_samples,)
            The class labels.


        Returns
        -------
        self : returns a trained decision tree

        layout : tree structure

        optimal_data : dictionary of data labels for each optimal split

        suboptimal_data : dictionary of data labels in the suboptimal splits.
        """
        self.X_train = X
        self.y_train = y

        if self.method == 'Classification':

            # Generate the tree structure and the direction of data splits
            layout, optimal_data, suboptimal_data = decision_tree_layout_classification(
                self.X_train, self.y_train, self.splitting_method)

            # Store for predictions
            self.layout = layout
            self.optimal_data = optimal_data
            self.suboptimal_data = suboptimal_data

        elif self.method == 'Regression':

            # Generate the tree structure and the direction of data splits
            layout, optimal_data, suboptimal_data = decision_tree_layout_regression(
                self.X_train, self.y_train, self.splitting_method)

            # Store for predictions
            self.layout = layout
            self.optimal_data = optimal_data
            self.suboptimal_data = suboptimal_data

    def predict(self, X_test):
        """ 
        Makes predictions on the testing data.

        Parameters
        ----------
        X : dataframe, shape (n_samples, n_features)
            The input data.


        Returns
        -------
        labels: an array of predicted labels
        """

        if self.method == 'Classification':

            # Generate the predictions
            labels = decision_tree_predict_classification(
                self.X_train, self.y_train, X_test, self.layout, self.optimal_data, self.suboptimal_data)

        elif self.method == 'Regression':

            # Generate the predictions
            labels = decision_tree_predict_regression(
                self.X_train, self.y_train, X_test, self.layout, self.optimal_data, self.suboptimal_data)

        return labels
