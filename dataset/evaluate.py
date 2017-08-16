# Origin: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
# Edited: Duino Du

import numpy as np

def _confusion_matrix(label_pred, label_true, n_class):
    """compute confusion matrix

    Args:
        label_pred (np.ndarray, N): predict label
        label_true (np.ndarray, N): groundtruth label
        n_class (int): number of classes 

    Returns: 
        hist (np.ndarray, n_class x n_class)
    """
    mask = (label_true >= 0) & (label_true < n_class)
    tmp = n_class * label_true[mask].astype(int) + label_pred[mask]
    hist = np.bincount(tmp, minlength=n_class**2).reshape(n_class, n_class)
    return hist

def evaluate(predicts, groundtruths, n_class):
    """Return accuracy score evaluation result

    Args:
        predicts (list[np.ndarray]): list of predict labels
        groundtruths (list[np.ndarray]): list of groundtruth labels
        n_class (int): TODO

    Returns:
        (dict): for print
        cls_iou(dict): iou for each class

    """
    hist = np.zeros((n_class, n_class))
    for pred, label in zip(predicts, groundtruths):
        hist += _confusion_matrix(pred.flatten(), label.flatten(), n_class)

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)

    freq = hist.sum(axis=1) / hist.sum()
    freq_acc = (freq[freq>0] * iou[freq>0]).sum()

    cls_iou = dict(zip(range(n_class), iou))
    
    return {'Overall Acc: \t':acc,
            'Mean Acc: \t': acc_cls,
            'Freq Acc: \t': freq_acc,
            'Mean IoU: \t': mean_iou}, cls_iou
