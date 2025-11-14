# evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np


def plot_confusion(y_true, y_pred, labels=None, figsize=(6, 5)):
    """
    Plot a confusion matrix using seaborn heatmap.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        List of class labels (order is respected).
        If None, labels will be inferred from y_true and y_pred.
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object
    """
    if labels is None:
        # auto infer unique labels
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_roc(y_true, y_scores, pos_label=None, figsize=(6, 5)):
    """
    Plot ROC curve for binary classification.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_scores : array-like
        Target scores (probabilities or decision function)
    pos_label : str or int, optional
        Label considered as "positive class". If None, taken from y_true.
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object
    auc : float
        ROC AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color="blue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig, auc
