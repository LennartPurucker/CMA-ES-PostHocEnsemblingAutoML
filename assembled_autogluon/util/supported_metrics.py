from sklearn.metrics import roc_auc_score
from autogluon.core.metrics import make_scorer, balanced_accuracy, roc_auc


def msc(metric_name, is_binary):
    if metric_name == "balanced_accuracy":
        return balanced_accuracy
    elif metric_name == "roc_auc":
        if is_binary:
            return roc_auc
        else:
            return make_scorer('roc_auc_ovr_macro',
                               roc_auc_score,
                               multi_class='ovr',
                               average='macro',
                               greater_is_better=True,
                               needs_proba=True,
                               needs_threshold=False)
    else:
        raise ValueError("Unknown metric name: {}".format(metric_name))
