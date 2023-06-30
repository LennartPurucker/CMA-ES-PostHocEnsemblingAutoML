import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "../../.."))  # have to do this because of singularity. I hate it

from pathlib import Path

from assembled.metatask import MetaTask
from assembled.ensemble_evaluation import evaluate_ensemble_on_metatask

from assembled_ensembles.util.config_mgmt import get_ensemble_switch_case, get_ensemble_switch_case_config, \
    get_basic_ensemble_names
from assembled_ensembles.default_configurations.supported_metrics import msc
import time
from assembled_ensembles.util.config_mgmt import get_all_ensemble_names
from assembled_ensembles.configspaces.evaluation_parameters_grid import get_config_space, get_name_grid_mapping
from ConfigSpace import Configuration

import random

if __name__ == "__main__":
    benchmark_name = "benchmark_0"
    pruner = "SiloTopN"  # "SiloTopN" "TopN"
    folds_to_run_on = [0]

    ens_list = []
    for ensemble_method_name in [
        "CMA-ES|batch_size:dynamic|normalize_weights:softmax|trim_weights:ges-like-raw",
        "CMA-ES|batch_size:25|normalize_weights:no|start_weight_vector_method:average_ensemble",
        "CMA-ES|batch_size:25|normalize_weights:no",
        "CMA-ES|batch_size:25|normalize_weights:no|sigma0:2",
        "CMA-ES|batch_size:25|normalize_weights:softmax|trim_weights:ges-trim-round"
    ]:
        cs = get_config_space("ENSW", small=True)
        name_grid_mapping = get_name_grid_mapping("ENSW")
        config = Configuration(cs, name_grid_mapping[ensemble_method_name])
        cs.check_configuration(config)
        ens_list.append(config)
    print(ens_list)

    # cs, cs_grid = get_config_space("ENSW", small=True, return_grid=True)
    # ens_list = list(cs_grid)  # random.sample(cs_grid, 10)
    # # ens_list = ["num_solver.DE.PositiveBounded", "num_solver.CMAES.Unbounded"]
    # # ens_list = get_basic_ensemble_names()

    for openml_task_id in ["146820"]:
        # Dataa:  "168784", "146820"
        print("1############################# TASK", openml_task_id)
        # Default to -1 as many ensemble methods only can use -1 and nothing else.
        n_jobs = -1

        # -- Build Paths
        file_path = Path(os.path.dirname(os.path.abspath(__file__)))
        tmp_input_dir = file_path.parent / "benchmark" / "output" / benchmark_name / "task_{}/final_output/{}".format(
            openml_task_id, pruner)

        # -- Rebuild The Metatask
        mt = MetaTask()
        mt.read_metatask_from_files(tmp_input_dir, openml_task_id)

        for metric_name in ["roc_auc"]:  # , "log_loss", "f1", "roc_auc", "balanced_accuracy"
            print("2############################# METRIC", metric_name)
            for idx, ensemble_method_name in enumerate(ens_list, 1):
                print(f"3############################# ENSEMBLE {idx}/{len(ens_list)}", ensemble_method_name)
                # -- Setup Evaluation variables
                # Get the metric(s)
                is_binary = len(mt.class_labels) == 2
                # If the ensemble requires the metric, we assume the labels to be encoded
                ens_metric = msc(metric_name, is_binary, list(range(mt.n_classes)))
                # For the final score, we need the original labels
                score_metric = msc(metric_name, is_binary, mt.class_labels)

                predict_method = "predict_proba" if ens_metric.requires_confidences else "predict"

                # -- Get techniques (initialize new for each task due to randomness and clean start)

                if isinstance(ensemble_method_name, str):
                    rng_seed = 3151278530
                    technique_run_args = get_ensemble_switch_case(ensemble_method_name, rng_seed=rng_seed,
                                                                  metric=ens_metric, n_jobs=n_jobs,
                                                                  is_binary=is_binary, labels=list(range(mt.n_classes)))
                else:
                    rng_seed = cs.meta["rng_seed"] if folds_to_run_on is None \
                        else cs.meta["seed_function_individual_fold"](cs.meta["rng_seed"], folds_to_run_on[0])
                    technique_run_args = get_ensemble_switch_case_config(ensemble_method_name,
                                                                         rng_seed=rng_seed,
                                                                         metric=ens_metric, n_jobs=n_jobs,
                                                                         is_binary=is_binary,
                                                                         labels=list(range(mt.n_classes)))

                st = time.time()
                scores = evaluate_ensemble_on_metatask(mt, technique_name=ensemble_method_name, **technique_run_args,
                                                       save_evaluation_metadata=True,
                                                       return_scores=score_metric, folds_to_run=folds_to_run_on,
                                                       use_validation_data_to_train_ensemble_techniques=True,
                                                       verbose=True, isolate_ensemble_execution=False,
                                                       predict_method=predict_method,
                                                       store_metadata_in_fake_base_model=False)
                print("Time Taken: ", time.time() - st)
                print(scores)
                print("K-Fold Average Performance:", sum(scores) / len(scores))
