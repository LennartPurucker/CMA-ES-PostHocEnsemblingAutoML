def transpose_means(data):
    re_data = data.pivot(index="dataset_name", columns="Ensemble Technique")
    re_data.columns = re_data.columns.droplevel(0)
    return re_data


def get_mean_over_cross_validation(fold_performance_data, metric_name, select_only_basic=True):
    df = fold_performance_data.drop(columns=["Fold"]).groupby(
        by=["dataset_name", "Ensemble Technique", "Setting", "TaskID"]).aggregate("mean").reset_index()

    if select_only_basic:
        return df[["dataset_name", "Ensemble Technique", metric_name]]
    else:
        return df

def get_std_over_cross_validation(fold_performance_data, metric_name, select_only_basic=True):
    df = fold_performance_data.drop(columns=["Fold"]).groupby(
        by=["dataset_name", "Ensemble Technique", "Setting", "TaskID"]).aggregate("std").reset_index()

    if select_only_basic:
        return df[["dataset_name", "Ensemble Technique", metric_name]]
    else:
        return df

def normalize_performance(ppd, baseline_algorithm, higher_is_better):
    def normalize_function(row):
        # https://stats.stackexchange.com/a/178629 scale to -1 = performance of baseline to 0 = best performance

        if not higher_is_better:
            tmp_row = row.copy() * -1
        else:
            tmp_row = row.copy()

        baseline_performance = tmp_row[baseline_algorithm]
        range_fallback = abs(tmp_row.max() - baseline_performance) == 0

        if range_fallback:
            mask = abs(tmp_row - baseline_performance) == 0
            tmp_row[~mask] = -10  # par10 like

            tmp_row[mask] = -1
            return tmp_row

        res = (tmp_row - baseline_performance) / (tmp_row.max() - baseline_performance) - 1

        return res

    return ppd.apply(normalize_function, axis=1)
