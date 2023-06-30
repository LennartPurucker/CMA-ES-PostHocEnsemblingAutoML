# Assembled-AutoGluon: Build Metatasks from the Search Results of AutoGluon

## Installation

To run this part of our code, you will require a specific environment.
We suggest to use [Docker](https://www.docker.com/) or [Singularity](https://sylabs.io/docs/).
Once you have set up one of the above, use the dockerfile or singularity def file in the `environment` directory to
build the python environment required to run the following scripts.

Alternatively, you can install the requirements by running the following commands in a (virtual) Python environment. 
Sadly, we can not provide a `requirements.txt` for this step, as it does not support `--no-dependencies`. 
Please be aware that the code was only tested for Python 3.8 on Linux! 
Moreover, make sure to have the newest version of pip. 

```bash 
python -m pip install -U pip && pip install -U setuptools wheel
python -m pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchtext==0.13.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
python -m pip install autogluon==0.6.2
python -m pip install --no-cache-dir --no-dependencies assembled[openml]==0.0.4
python -m pip install --no-cache-dir numpy pandas scikit-learn scipy tables openml requests
```

## Advanced Details

The files `../conf/bacc_benchmark_data.json`, `../conf/roc_auc_benchmark_data.json` contain
the exact parameters we have supplied to the scripts below.
As reproducing the exact data requires a lot of time (4 hours per fold per dataset) and a lot of memory
(32-126 GB per fold per dataset), we provide a minimal example that shows that our code works.

We limited the number of CPUs and memory outside the scripts using [SLURM](https://slurm.schedmd.com/documentation.html)
.
You could also limit the CPUs and memory with Linux, Docker, or Singularity.
For the minimal example we ignore this and default to using all available CPUs and memory.

## Minimal Example

Thought the minimal example we will use "-1" as an OpenML task ID. This will adjust the `time_limit` to be in minutes
instead of hours and use a small toy dataset from sklearn.

### 1. Run AutoGluon on the dataset and parse the base models' prediction data

First run the following command in the CLI while being in the docker/singularity environment or start the scripts with
an IDE and the container as interpreter.
This wil run AutoGluon on each fold of the toy dataset for 1 minute wit 1 GB of RAM and all available CPU cores.
Thereby, AutoGluon optimizes for the metric balanced accuracy.
Additionally, this will parse and save the prediction data of the base models.
The output is saved under `benchmark/output/minimal_example`.

```shell
python run_ag_on_metatask.py -1 1 0,1,2,3,4,5,6,7,8,9 balanced_accuracy minimal_example
```

### 2. Save the Prediction Data

Finally, we build the final data structure (a Metatask from the Assembled Framework), which is later used for evaluating
the ensemble methods.

```shell
python run_full_metatask_builder.py -1 1 balanced_accuracy hdf minimal_example
```

## Detail Usage Documentation

To build a metatask with AutoGluon, you have to execute the following scripts in order.
You have to do this for each metric and all the details in the metric specific files (MSF)
(e.g., `../conf/bacc_benchmark_data.json`, `../conf/roc_auc_benchmark_data.json`)
as noted below to fully reproduce all the data for all base models.

1) `python run_ag_on_metatask.py task_id time_limit folds_to_run metric_name benchmark_name`
    * `task_id`: an OpenML task ID (for testing, pass `-1`)
      * see `../conf/task_data.json` for all used task IDs.
    * `time_limit`: the time limit / search time for auto-sklearn in hours
      * see MSF: "framework_time_hours"
    * `folds_to_run`: the folds to run (for parallelization). Either a list of a subset of folds or all folds,
      e.g, `0,1,2,3,4,5,6,7,8,9`. Alternatively, only one fold, e.g., `0`.
    * `metric_name`: metric name of the metric to be optimized by ask, we expect the import name of the metric
      * see MSF: "framework_extras/metric"
    * `benchmark_name`: the name of the folder where the results are supposed to be stored. (
      Under `benchmark/output/{{benchmark_name}}`)

2) `python run_full_metatask_builder.py task_id time_limit metric_name file_format benchmark_name`
    * `task_id`: an OpenML task ID (for testing, pass `-1`)
    * `time_limit`: the time limit / search time for auto-sklearn in hours (needed for metadata)
      * see MSF: "framework_time_hours"
    * `metric_name`: metric name of the metric to be optimized by ask, we expect the import name of the metric
      * see MSF: "framework_extras/metric"
    * `file_format`: Allowed formats are {"csv", "hdf", "feather"}. 
      * We suggest to default to HDF!
    * `benchmark_name`: the name of the folder where the results are supposed to be stored. (
      Under `benchmark/output/{{benchmark_name}}`)
      * The same as for the previous step.

## Notes

* AutoGluon does not support a memory limit. It uses all available memory. Hence, we have to limit this in the outer
  scope (e.g., SLURM).
* If you parallelize: You first have to run `run_ag_on_metatask.py` for all folds before
  running `run_full_metatask_builder.py`.


