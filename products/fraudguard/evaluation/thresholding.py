import numpy as np
from sklearn.metrics import precision_recall_curve


def find_threshold_for_recall(
    y_true,
    y_scores,
    min_recall: float = 0.70,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    valid_indices = np.where(recall[:-1] >= min_recall)[0]

    if len(valid_indices) == 0:
        return 0.5

    best_index = valid_indices[np.argmax(precision[valid_indices])]

    return float(thresholds[best_index])
