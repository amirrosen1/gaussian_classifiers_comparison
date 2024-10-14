import numpy as np


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    # Calculate the number of misclassifications
    misclassified = np.sum(y_true != y_pred)

    if normalize:
        # Normalize by the number of samples
        return misclassified / len(y_true)
    else:
        # Return the number of misclassifications
        return misclassified


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    correct_predictions = np.sum(y_true == y_pred)
    accuracy_score = correct_predictions / len(y_true)
    return accuracy_score
