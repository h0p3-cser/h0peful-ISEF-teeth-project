"""Leave-one-tooth-out partitions and group-level classification metrics."""
import json

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)


def tooth_folds(groups, states):
    teeth = groups.tooth_id.to_numpy()
    splits = []
    for held in sorted(set(teeth)):
        train = np.flatnonzero(teeth != held)
        test = np.flatnonzero(teeth == held)
        if set(groups.iloc[train].state_id) != set(states):
            raise ValueError(f"Leaving out {held} removes a required training stage")
        splits.append((held, train, test))
    return splits


def metrics(y, prob, states):
    y = np.asarray(y)
    prob = np.asarray(prob)
    if prob.shape != (len(y), len(states)) or not np.isfinite(prob).all():
        raise ValueError("Expected one finite probability per stage and observation")
    np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-7)
    labels = list(range(len(states)))
    pred = prob.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=labels)
    stages = []
    for i, state in enumerate(states):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        binary = y == i
        auc = float(roc_auc_score(binary, prob[:, i])) if binary.any() and not binary.all() else None
        ap = float(average_precision_score(binary, prob[:, i])) if binary.any() else None
        stages.append(dict(state_id=state, support=tp+fn, true_positive=tp, false_negative=fn,
                           false_positive=fp, true_negative=tn, precision=tp/(tp+fp) if tp+fp else 0.0,
                           precision_defined=bool(tp+fp), sensitivity=tp/(tp+fn) if tp+fn else None,
                           specificity=tn/(tn+fp) if tn+fp else None, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                           roc_auc_ovr=auc, average_precision=ap))
    aucs = [s['roc_auc_ovr'] for s in stages]
    overall = dict(correct=int((y == pred).sum()), total=len(y), accuracy=float(accuracy_score(y, pred)),
                   balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                   macro_f1=float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
                   roc_auc_ovr_macro=float(np.mean(aucs)) if None not in aucs else None,
                   log_loss=float(log_loss(y, prob, labels=labels)), confusion_matrix=json.dumps(cm.tolist()))
    return overall, stages
