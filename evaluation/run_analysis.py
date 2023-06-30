from evaluation.analysis_data import read_data
from evaluation import analysis_utils
from evaluation import evaluations
import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


# -- Section 2
def _rank_overfitting_tabel(ranks_per_task_metric, postifx):
    col_names = ["Method", "Metric", "Task Type", "Validation - Mean Rank", "Test - Mean Rank",
                 "Absolute Rank Change Validation to Test"]
    rows = []
    for metric, task_type, val_mean_r, test_mean_r, val_r, test_r in ranks_per_task_metric:
        for v_mean_r, t_mean_r, v_r, t_r, method in zip(val_mean_r, test_mean_r, val_r, test_r, test_r.index):
            rows.append((method, metric, task_type, v_mean_r, t_mean_r, f"{v_r} -> {t_r}"))

    df = pd.DataFrame(rows, columns=col_names)
    df.to_csv(f"./out/overfitting_report_{postifx}.csv", index=False)


def standard_eval(results_per_task, metric_data, rename_and_default_map, extra_plot_postfix="", save_plots=False):
    ranks = []
    for task_type, ppf in results_per_task:
        print("\n\nNumber of Dataset for {} {} {}: {}".format(task_type, metric_data["metric_name"], extra_plot_postfix,
                                                              len(set(ppf["dataset_name"].tolist()))))
        # Preprocess
        ppf = ppf[ppf["Ensemble Technique"].isin(list(rename_and_default_map.keys()))]
        ppf["Ensemble Technique"] = ppf["Ensemble Technique"].apply(lambda x: rename_and_default_map[x])
        plot_postfix = "for Task Type {} and Metric {} {}".format(task_type, metric_data["metric_name"],
                                                                  extra_plot_postfix)

        # -- CD Plot
        perf_pd = analysis_utils.transpose_means(
            analysis_utils.get_mean_over_cross_validation(ppf, metric_data["metric_name"]))

        if extra_plot_postfix == "(Final)":
            std_pd = analysis_utils.transpose_means(analysis_utils.get_std_over_cross_validation(ppf, metric_data["metric_name"]))

            overview_pd = perf_pd.round(4).astype(str) + " (± " + std_pd.round(4).astype(str) + ")"

            for index_name, col_name in zip(perf_pd.index, list(perf_pd.idxmax(axis=1))):
                overview_pd.loc[index_name, col_name] = r"\textbf{" + overview_pd.loc[index_name, col_name] + "}"
            overview_pd.to_csv(f"./out/performance_overview_{metric_data['metric_name']}_{task_type}.csv")


        evaluations.cd_evaluation(perf_pd, metric_data["maximize_metric"], plot_postfix, plot=True)

        # -- Normalized Improvement Plot
        normalized_perf_pd = analysis_utils.normalize_performance(perf_pd, "SingleBest",
                                                                  metric_data["maximize_metric"])
        evaluations.normalized_improvement_boxplot(normalized_perf_pd, "SingleBest", plot_postfix)

        # -- Overfitting Report
        ppf["validation_score"] = ppf["validation_loss"].apply(metric_data["loss_to_score"])
        val_perf_pd = analysis_utils.transpose_means(
            analysis_utils.get_mean_over_cross_validation(ppf, "validation_score"))
        assert list(perf_pd) == list(val_perf_pd)
        assert list(perf_pd.index) == list(val_perf_pd.index)
        # Remove stacking because we have no validation data for it
        if "Stacking" in list(val_perf_pd):
            val_perf_pd = val_perf_pd.drop(columns=["Stacking", "CMA-ES"])
            perf_pd = perf_pd.drop(columns=["Stacking", "CMA-ES"])
        val_r = val_perf_pd.rank(axis=1, ascending=False).mean()
        test_r = perf_pd.rank(axis=1, ascending=False).mean()
        ranks.append((metric_data["metric_name"], task_type, val_r, test_r, val_r.rank(), test_r.rank()))

        # -- Ensemble Size
        print("Ensemble Size Average:", analysis_utils.transpose_means(
            analysis_utils.get_mean_over_cross_validation(ppf, "ensemble_size")).mean())

    return ranks


def default_application(results_per_task, metric_data):
    # -- Get results for straightforward application
    rename_and_default_map = {
        "TopN.CMA-ES|batch_size:dynamic|normalize_weights:no": "CMA-ES",
        "TopN.SingleBest": "SingleBest",
        "TopN.EnsembleSelection|use_best": "GES",
    }

    ranks = standard_eval(results_per_task, metric_data, rename_and_default_map,
                          extra_plot_postfix="(Default Application)")

    return ranks


# -- Section 3
def weight_normalization_overview(results_per_task, metric_data):
    rename_and_default_map = {
        "TopN.CMA-ES|batch_size:dynamic|normalize_weights:no": "CMA-ES",
        "TopN.CMA-ES|batch_size:dynamic|normalize_weights:softmax|trim_weights:no": "CMA-ES-Softmax",
        "TopN.CMA-ES|batch_size:dynamic|normalize_weights:softmax|trim_weights:ges-like-raw": "CMA-ES-ImplicitGES",
        "TopN.CMA-ES|batch_size:dynamic|normalize_weights:softmax|trim_weights:ges-trim-round": "CMA-ES-ExplicitGES",
        "TopN.SingleBest": "SingleBest",
    }

    standard_eval(results_per_task, metric_data, rename_and_default_map, extra_plot_postfix="(Weight Normalization)")


# -- Final Evaluation
def final_overview(results_per_task, metric_data):
    rename_and_default_map = {
        "TopN.CMA-ES|batch_size:dynamic|normalize_weights:no": "CMA-ES",
        "TopN.CMA-ES|batch_size:dynamic|normalize_weights:softmax|trim_weights:ges-trim-round": "CMA-ES-ExplicitGES",
        "TopN.SingleBest": "SingleBest",
        "TopN.EnsembleSelection|use_best": "GES",
        "TopN.LogisticRegressionStacking": "Stacking"
    }

    ranks = standard_eval(results_per_task, metric_data, rename_and_default_map, extra_plot_postfix="(Final)")
    return ranks


def run(benchmark_name, eval_name):
    results_per_task, metric_data, baseline_algorithm = read_data(benchmark_name, eval_name)

    # -- Eval default application of CMA-ES
    default_ranks = default_application(results_per_task, metric_data)
    weight_normalization_overview(results_per_task, metric_data)
    final_ranks = final_overview(results_per_task, metric_data)

    return default_ranks, final_ranks


def _run():
    to_anal_data = [
        ("bacc", "ensemble_evaluations_enswf"),
        ("roc_auc", "ensemble_evaluations_enswf")
    ]

    Path("./out").mkdir(exist_ok=True)

    default_ranks = []
    final_ranks = []
    for bm_name, eval_name in to_anal_data:
        tmp_default_ranks, tmp_final_ranks = run(bm_name, eval_name)
        default_ranks.extend(tmp_default_ranks)
        final_ranks.extend(tmp_final_ranks)

    if default_ranks:
        _rank_overfitting_tabel(default_ranks, "detault")
    if final_ranks:
        _rank_overfitting_tabel(final_ranks, "final")


if __name__ == "__main__":
    _run()
