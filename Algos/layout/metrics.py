import numpy as np

def log_loss_mine(prediction_prob, true_value):
    """Calculates the log-loss of a classifier given predicted probabilities and true values."""
    eps = 1e-15
    prediction_prob = np.clip(prediction_prob, eps, 1-eps)
    N = prediction_prob.shape[0]
    tmp = -(true_value * np.log(prediction_prob) + (1-true_value)*np.log(1- prediction_prob))
    tmp = np.sum(tmp, axis = 0)/N
    loss = np.min(tmp)
    return loss