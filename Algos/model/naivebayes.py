#----------------- Packages
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

#----------------- Function
class myNaiveBayes:
    """Implementation of the Naive Bayes Classifier using numpy"""

    def __init__(self, method, alpha):
        self.method = method
        self.alpha = alpha

    def fit(self, X, y):
        # Fit function
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        if self.method == 'Gaussian':
            # No smoothing factor present in GaussianNB
            self.alpha = None

            # Number of unique classes
            num_classes = np.unique(self.y)
            self.classes = num_classes

            # Temporary dataframe for appending X_train and y_train
            tmp_df = pd.DataFrame(data=self.X)
            tmp_df['Output'] = self.y

            # Calculate the prior, and frequency of each class
            d_prior = {}
            d_mean = {}
            d_std = {}
            for i in self.classes:
                # Subset the data for each class
                X_class = tmp_df[tmp_df['Output'] == i]

                # Subset the column indices to only include the features
                X_class = X_class.iloc[:, 0: len(tmp_df.columns) - 1]

                # Get the prior probability
                prior = X_class[0].count() / len(tmp_df)
                d_prior[i] = prior

                # Get the mean of each column
                d_mean[i] = np.mean(X_class)

                # Get the std of each column
                d_std[i] = np.std(X_class)

            # Store the prior, mean, and std values for predictions
            self.prior = d_prior
            self.mean = d_mean
            self.std = d_std

        elif self.method == 'Multinomial':
            # Fit a count vectorizer to our training set
            cv = CountVectorizer(encoding='latin-1',
                                 strip_accents='ascii', analyzer='word')
            self.cv = cv

            # Create counts vector for training set
            counts = self.cv.fit_transform(X)

            # Create dataframe of counts vector
            df_counts = pd.DataFrame(
                counts.A, columns=self.cv.get_feature_names(), index=X.index)
            df_counts['Output'] = y

            # Number of unique classes
            num_classes = np.unique(y)
            self.classes = num_classes

            d_prob = {}
            # Create nested dictionary
            for i in self.classes:
                tmp = i
                d_prob[i] = {}

            copy = d_prob
            class_prior_dict = {}
            for i, j in zip(self.classes, d_prob):
                # Subset the data for each class
                tmp_class = df_counts[df_counts['Output'] == i]

                # Get the class prior for each class
                class_prior = len(tmp_class)/len(df_counts)
                class_prior_dict[i] = class_prior

                # Record the counts of the class
                counts_class = np.array(
                    tmp_class.sum()[0: len(tmp_class.columns) - 1].values)

                # Calculate the probability of the class
                sum_features_class = counts_class + self.alpha
                total_sum_class = np.sum(
                    counts_class, axis=0) + len(tmp_class[0: len(tmp_class.columns) - 1])
                probability_class = sum_features_class / total_sum_class

                # Store the probabilities into a dictionary
                column_indices = [x for x in range(0, len(tmp_class.columns))]

                # Create a temporary dict object to store the probabilities in and put them into our nested dict
                tmp_dict = {}
                for k, l in zip(column_indices, probability_class):
                    tmp_dict[k] = l
                    copy[i] = tmp_dict

            # Set our initial dict to our copy
            d_prob = copy
            self.prob_class = d_prob
            self.class_prior = class_prior_dict

    def predict(self, X):
        # Returns the predicted labels
        if self.method == 'Gaussian':

            # Calculate the probability of the predictions
            predict_proba = []
            for x_pred in X:
                for i, j, k in zip(self.mean.values(), self.std.values(), self.prior.values()):
                    # Calculate the P(X|Y)
                    p_x_y = np.array(1/(np.sqrt(2*np.pi*j**2))
                                     * np.exp(-((x_pred - i)**2)/(2*j**2))).prod()

                    # Multiply the prior with P(X|Y)
                    p_c_y = p_x_y * k
                    predict_proba.append(p_c_y)

            predict_proba = np.array(predict_proba)
            predict_proba = predict_proba.reshape((len(X), len(self.classes)))
            self.predict_proba = predict_proba

            # Select the maximum probability class label
            y_pred = []
            for i in predict_proba:
                # Find the index with the maximum value in the array
                tmp = (np.where(i == np.amax(i)))
                y_pred.append(tmp[0][0])

            y_pred = np.array(y_pred)
            return y_pred

        elif self.method == 'Multinomial':
            # Predictions
            pred_counts = self.cv.transform(X)

            # DF Counts
            df_pred_counts = pd.DataFrame(
                pred_counts.A, columns=self.cv.get_feature_names(), index=X.index)

            # Create a List for each row of testing set for each feature present (Indices)
            pred_features = []
            for i in df_pred_counts.values:
                features_ = np.where(i >= 1)[0]
                pred_features.append(features_)

            self.predict_proba = {}
            # Create nested dictionary
            for i in self.classes:
                tmp = i
                self.predict_proba[i] = {}

            copy = self.predict_proba
            # Iterate through each testing sample and create a probability vector of all the features present
            for i, j in zip(self.classes, self.predict_proba):
                tmp_dict = {}
                for k, l in zip(pred_features, range(0, len(pred_features))):
                    tmp_list = []
                    for m in k:
                        # For feature vector get the class probability
                        tmp = self.prob_class[i][m]
                        tmp_list.append(tmp)

                    # For each testing set sample, set the collection of probabilities
                    tmp_dict[l] = tmp_list

                # Now, obtain the collection of probabilities for the classes
                copy[i] = tmp_dict

            # Initialize prediction probabilitys from the copied dict object
            self.predict_proba = copy

            # Calculate the product of the probabilities in each testing sample
            for i in self.predict_proba:
                for j in self.predict_proba[i]:
                    # Multiply the class prior with the class probabilities
                    tmp_class_prior = np.array(self.class_prior[i])
                    tmp = np.append([self.predict_proba[i][j]], [
                                    tmp_class_prior])
                    self.predict_proba[i][j] = np.prod(tmp)

            # We want to append both class values for each testing sample
            # Create a list object for each dict key
            total_dict = {k: [] for k in range(len(pred_features))}
            for i in self.predict_proba.keys():
                tmp = self.predict_proba[i]
                for j in tmp:
                    test_ = self.predict_proba[i][j]
                    total_dict[j].append(test_)

            self.predict_proba = total_dict

            # Compare the class probabilities to select the label
            tmp = self.predict_proba.copy()
            tmp_list = []
            for i in tmp:
                # Convert predict_proba to a list of arrays
                tmp_list.append(tmp[i])

                # Classifies the prediction samples
                tmp[i] = np.where(tmp[i] == np.amax(tmp[i]))

            # Converted the dictionary to a list of probabilities
            self.predict_proba = tmp_list

            # Normalize probabiltiies
            normalized_list = []
            for i in self.predict_proba:
                sum_ = np.sum(i)
                normalized_list.append(i/sum_)

            self.predict_proba = normalized_list
            self.predict_proba = np.array(self.predict_proba)

            # Create a list of y_pred labels from the dictionary
            y_pred = []
            for i in tmp:
                y_pred.append(tmp[i][0][0])

            # Rename discrete values with text-based classes
            y_pred = [self.classes[0] if x == 0 else self.classes[1]
                      for x in y_pred]
            y_pred = np.array(y_pred)

        return y_pred
