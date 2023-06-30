import pandas as pd
import json


def _handle_nan_values(performance_per_fold_per_dataset, baseline_algorithm):
    print(f"Found {performance_per_fold_per_dataset.isna().sum()} nan values")
    performance_per_fold_per_dataset[['validation_loss', 'ensemble_size']] = performance_per_fold_per_dataset[
        ['validation_loss', 'ensemble_size']].fillna(value=-1)
    # Set SingleBest Size to 1
    performance_per_fold_per_dataset.loc[
        performance_per_fold_per_dataset["Ensemble Technique"] == baseline_algorithm, "ensemble_size"] = 1
    # Fill remaining nan values with 0
    performance_per_fold_per_dataset = performance_per_fold_per_dataset.fillna(value=0)
    return performance_per_fold_per_dataset


def _preprocess_ensemble_names(performance_per_fold_per_dataset):
    idea_map_methods = {"SingleBest": "SingleBest", "EnsembleSelection": "GES", "CMA-ES": "CMA-ES",
                        "LogisticRegressionStacking": "Stacking"}
    fallback_name = "Other"

    def name_mapping(row):
        name = row[0]
        prefix = name.split(".", 1)[0] + "."
        config_name = name.split(".", 1)[1]

        if "|" in config_name:
            method_name = config_name.split("|", 1)[0]
        else:
            method_name = config_name

        if method_name in idea_map_methods:
            return name, idea_map_methods[method_name]

        # Fall back case for unknown method
        return name, method_name

    performance_per_fold_per_dataset[["Ensemble Technique", "Method"]] = performance_per_fold_per_dataset[
        ["Ensemble Technique", "Fold"]].apply(name_mapping, axis=1, result_type="expand")

    return performance_per_fold_per_dataset


def _get_metric_data(benchmark_name):
    with open(f"../conf/{benchmark_name}_benchmark_data.json") as f:
        metric_name = json.load(f)["framework_extras"]["metric"]

    return {"metric_name": metric_name,
            # works since all metrics we use are to be maximized
            "maximize_metric": True,
            # Transform loss to score; works since all metrics we use have 1 as optimum
            "loss_to_score": lambda x: 1 - x if x >= 0 else -1
            }


def _get_results_per_task(performance_per_fold_per_dataset):
    binary = performance_per_fold_per_dataset[performance_per_fold_per_dataset["n_classes"] == 2]
    multi = performance_per_fold_per_dataset[performance_per_fold_per_dataset["n_classes"] != 2]

    results_per_task = [
        ("Binary Classification", binary),
        ("Multi-class Classification", multi),
    ]

    return results_per_task


def read_data(benchmark_name, eval_name):
    performance_per_fold_per_dataset = pd.read_csv(f"./tmp_results/{benchmark_name}_{eval_name}_fold_results.csv")

    # -- Preprocess the names
    baseline_name = "SingleBest"
    performance_per_fold_per_dataset = _preprocess_ensemble_names(performance_per_fold_per_dataset)

    # -- Define what algorithms to evaluate
    baseline_algorithm = f"TopN.{baseline_name}"

    # -- Preprocess the data
    performance_per_fold_per_dataset = _handle_nan_values(performance_per_fold_per_dataset, baseline_algorithm)

    # - Add task metadata
    performance_per_fold_per_dataset = performance_per_fold_per_dataset.merge(
        pd.read_csv(f"./tmp_results/{benchmark_name}_metatask_analysis_results.csv"),
        left_on=["TaskID", "Setting"], right_on=["TaskID", "Setting"],
        how="inner", validate="many_to_one"
    )

    # - Define evaluation metric
    metric_data = _get_metric_data(benchmark_name)

    # - Split data per task type
    results_per_task = _get_results_per_task(performance_per_fold_per_dataset)

    return results_per_task, metric_data, baseline_algorithm
