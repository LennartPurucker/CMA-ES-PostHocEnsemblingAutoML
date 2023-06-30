# Assembled-Ensembles

This directory contains the ensemble methods as well as code to enable their usage.
To evaluate the ensemble method we use the `assemlbed` framework as part of
the `run_evaluate_ensemble_on_metatask.py` script.

In detail, code for methods can be found in the `methods` directory.
The invocation of these methods by their configurations can be found in the `default_configurations` directory.
The supported configurations and the grid of hyperparameters can be found in the `configspaces` directory.
The `util` directory contains several important utilities necessary to run some ensemble techniques.
The `wrappper` directory contains the code that wraps around all ensemble methods to expose a standardized sklearn-like
interface.

## Installation

To run this part of our code, you will require a specific environment.
We suggest to use [Docker](https://www.docker.com/) or [Singularity](https://sylabs.io/docs/).
Once you have set up one of the above, use the dockerfile or singularity def file in the `environment` directory to
build the python environment required to run the following scripts.

Alternatively, you can install the requirements using the `requirements.txt` in the `environment` directory.
However, please be aware that the code was only tested for Python 3.8 on Linux! 

## Minimal Example

The following commands are examples to run each method once.
Thereby, we run it on base model data generated for a toy dataset from sklearn (TaskID "-1"); stored in
this repository.

```shell
python run_evaluate_ensemble_on_metatask.py -1 TopN "SingleBest" balanced_accuracy minimal_example_ens ensemble_evaluations_enswf no delayed -1 conf_0
python run_evaluate_ensemble_on_metatask.py -1 TopN "LogisticRegressionStacking" balanced_accuracy minimal_example_ens ensemble_evaluations_enswf no delayed -1 conf_1
python run_evaluate_ensemble_on_metatask.py -1 TopN "EnsembleSelection|use_best" balanced_accuracy minimal_example_ens ensemble_evaluations_enswf no delayed -1 conf_2
python run_evaluate_ensemble_on_metatask.py -1 TopN "CMA-ES|batch_size:dynamic|normalize_weights:no" balanced_accuracy minimal_example_ens ensemble_evaluations_enswf no delayed -1 conf_3
python run_evaluate_ensemble_on_metatask.py -1 TopN "CMA-ES|batch_size:dynamic|normalize_weights:softmax|trim_weights:no" balanced_accuracy minimal_example_ens ensemble_evaluations_enswf no delayed -1 conf_4
python run_evaluate_ensemble_on_metatask.py -1 TopN "CMA-ES|batch_size:dynamic|normalize_weights:softmax|trim_weights:ges-like-raw" balanced_accuracy minimal_example_ens ensemble_evaluations_enswf no delayed -1 conf_5
python run_evaluate_ensemble_on_metatask.py -1 TopN "CMA-ES|batch_size:dynamic|normalize_weights:softmax|trim_weights:ges-trim-round" balanced_accuracy minimal_example_ens ensemble_evaluations_enswf no delayed -1 conf_6
```

The results of the runs are stored under `benchmark/output/minimal_example_ens` with the name "conf_0", "conf_1", etc..
One needs to parse and evaluate them to compute the results which we build our evaluation on.
Otherwise, one can use the output of the script to obtain scores immediately.

## Detail Usage Documentation

To evaluate an ensemble on a metatask, to execute the following script with the appropriate parameters.
See `conf/evaluation_data.json` for details on used parameters.

1) `python run_evaluate_ensemble_on_metatask.py task_id pruner ensemble_method_name metric_name benchmark_name evaluation_name isolate_execution load_method folds_to_run_on ens_save_name`
    * `task_id`: an OpenML task ID (for testing, pass `-1`)
    * `pruner`: "TopN" to find the correct path.
    * `ensemble_method_name`: Name of the ensemble method's configuration
        * see `configspaces/name_grid_mapping.json`
    * `metric_name`: metric name of the metric to be optimized by ask, we expect the import name of the metric
        * "roc_auc" or "balanced_accuracy"
    * `benchmark_name`: The name of the benchmark, used to find the correct base path.
        * The path to the output of the scripts from `assembled_autogluon`
    * `evaluation_name`: The name of the evaluation setup, used to find the correct base path.
        * e.g., "ensemble_evaluations_enswf"
    * `isolate_execution`: If `yes`, we isolate the execution of the ensemble to avoid memory leakage (only works on
      linux)
    * `load_method`: If "delayed", we will only load data once needed for the specific fold. Otherwise, we will load the
      whole metatask into memory at the start.
        * We suggest to default to "delayed"!
    * `folds_to_run_on`: If "-1" all folds are run sequentially. Else the number corresponds to the fold on which the
      ensemble is evaluated.
        * See `special_cases/parallel` in `conf/evaluation_data.json` for the list of task ID where we executed each
          fold individually instead of sequentially. This is important as it changes the random seed!
    * `ens_save_name` The name used to identify this configuration in the output for parsing.