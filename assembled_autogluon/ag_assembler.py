import sys
import os
import glob
import logging
import pathlib
import pickle
import time
import hashlib
import numpy as np
import pandas as pd
from heapq import heappush, heappop
from shutil import rmtree
from autogluon.tabular import TabularPredictor

from sklearn.preprocessing import LabelEncoder

from typing import Union, Optional, List

from assembled.metatask import MetaTask

# -- Logging
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger("AgAssembler")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


class AgAssembler:
    """ Assembler to build Metatasks from AutoGluon Data

    Parameters
    ----------
    metatask: MetaTask
        Metatask for which we want to collect predictor data
    tmp_output_dir: str, Path
        Path to the dir where the output shall be stored
    folds_to_run: List[int], default=None
        Which outer folds of the metatak's split to run
    save_disc_space: bool, default=True
        If true, we delete already processed data from auto-gluon as much as possible.
    """

    def __init__(self, metatask: MetaTask, tmp_output_dir: Union[str, pathlib.Path],
                 folds_to_run: Optional[List[int]] = None, save_disc_space: bool = True):

        self.metatask = metatask
        self.tmp_output_dir = pathlib.Path(tmp_output_dir)
        self.folds_to_run = folds_to_run if folds_to_run is not None else [i for i in range(self.metatask.max_fold + 1)]
        self.save_disc_space = save_disc_space
        self.classification = True

    # -- Run AutoGluon
    def _verify_run_environment(self):
        # Verify clean env environment
        fold_dir_exists = all([self.tmp_output_dir.joinpath("fold_{}".format(f_idx)).exists()
                               for f_idx in self.folds_to_run])
        if self.tmp_output_dir.exists() and fold_dir_exists:
            raise ValueError("tmp_output_path {} and folds {} already exist. ".format(self.tmp_output_dir,
                                                                                      self.folds_to_run)
                             + "We wont delete it. Make sure to delete it yourself.")

    def run(self, metric_to_optimize, time_limit: int):
        """Run ASK on a Metatask and store the search results in a specific way

        Parameters
        ----------
        metric_to_optimize: Auto-Gluon Scorer
            Metric that is optimized
        time_limit: int
            Time in hours for the search
        """
        logger.info("Run AutoGluon on Data from Metatask")
        self._verify_run_environment()

        for iter_idx, (fold_idx, X_train, X_test, y_train, _) in enumerate(
                self.metatask._exp_yield_data_for_base_model_across_folds(self.folds_to_run), 1):
            logger.info("### Processing Fold {} | {}/{} ###".format(fold_idx, iter_idx, len(self.folds_to_run)))
            tmp_folder = self.tmp_output_dir.joinpath("fold_{}".format(fold_idx))

            # -- Setup AutoGluon

            if self.metatask.task_type == "classification":
                problem_type = "binary" if self.metatask.n_classes == 2 else "multiclass"
            else:
                raise ValueError("Not supported task type!")

            predictor = TabularPredictor(
                label=self.metatask.target_name,
                eval_metric=metric_to_optimize,
                path=tmp_folder,
                problem_type=problem_type,
                learner_kwargs=dict(cache_data=True)
            )

            # -- Build dataset and run AutoGluon
            train_data = pd.concat([X_train, y_train], axis=1)
            logger.info("Start Search")
            predictor.fit(
                time_limit=int(time_limit * 60 * 60),
                train_data=train_data,
                num_cpus="auto",
                presets="best_quality",
                calibrate=False,
                fit_weighted_ensemble=False,
            )

            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Search finished")
            logger.info(predictor.leaderboard())

            # -- Save results
            self._save_models_for_predictor(predictor, tmp_folder, fold_idx, X_train, y_train, X_test)

            # --- Fold Specific Info about prediction data
            self._clean_up_after_fold(tmp_folder)

        # --- Add Default Selection Constraints
        logger.info("Finished.")

    def _save_models_for_predictor(self, predictor, tmp_folder, fold_idx, X_train, y_train, X_test):
        logger.info("Start Saving Results...")

        # - Setup
        model_names = predictor.get_model_names()
        classes_ = np.unique(y_train)
        model_data = predictor.leaderboard()  # val score, fit and predict time on val
        ag_X_test = predictor.transform_features(X_test)

        # - Set up dir
        ag_dir = tmp_folder.joinpath(".ag_assembler")
        if not ag_dir.exists():
            os.mkdir(ag_dir)

        # - Save classes for current fold
        if (set(classes_) != set(predictor.class_labels)) and all(
                a == b for a, b in zip(classes_, predictor.class_labels)):
            raise ValueError("Classes of AutoGluon and Metatask do not match!",
                             "AutoGluon: ", predictor.class_labels, "Metatask: ", classes_)
        np.save(ag_dir.joinpath("classes_.npy"), classes_)

        # loop over all models
        for model_idx, model_name in enumerate(model_names, 1):
            logger.info(f"Process Model {model_name} | {model_idx}/{len(model_names)}")

            # -- Model specs
            spec_model_data = model_data[model_data["model"] == model_name]
            if spec_model_data.shape[0] != 1:
                raise ValueError("Multiple models with the same name!")
            spec_model_data = spec_model_data.iloc[0, :]

            bm_config = dict(
                model_name=model_name,
                stack_level=spec_model_data["stack_level"],
                score_val=spec_model_data["score_val"]
            )
            fit_time = spec_model_data["fit_time_marginal"]

            # - Predict and capture time needed
            st = time.time()
            # as_pandas true to guarantee correct order of classes
            test_y_pred = predictor.predict_proba(ag_X_test, model=model_name, as_pandas=True,
                                                  transform_features=False).to_numpy()
            predict_time = time.time() - st

            val_y_pred = predictor.get_oof_pred_proba(model=model_name, transformed=False,
                                                      train_data=X_train)
            val_indices = X_train.index
            # Sanity check
            if val_y_pred.shape[0] != X_train.shape[0]:
                raise ValueError("OOF data does not match train data!")
            if any(a != b for a, b in zip(list(val_y_pred.columns), classes_)):
                raise ValueError("Classes of Validation Predictions and Metatask do not match!")
            val_y_pred = val_y_pred.to_numpy()

            self._store_fold_predictors(fold_idx, model_idx, bm_config, val_y_pred, val_indices, test_y_pred,
                                        fit_time, predict_time, spec_model_data["fit_order"])

        logger.info("Finished Saving Results.")

    def _store_fold_predictors(self, fold_idx, model_idx, bm_config, val_y_pred, val_indices, test_y_pred,
                               fit_time, predict_time, model_evaluated_time):
        store_dir = self.tmp_output_dir.joinpath("fold_{}/.ag_assembler".format(fold_idx))
        predictor_dir = store_dir.joinpath("prediction_data")
        if not predictor_dir.exists():
            os.mkdir(predictor_dir)

        predictor_data = {
            "bm_config": bm_config,
            "val_y_pred": val_y_pred,
            "val_indices": val_indices,
            "test_y_pred": test_y_pred,
            "fit_time": fit_time,
            "predict_time": predict_time,
            "model_evaluated_time": model_evaluated_time
        }
        with open(predictor_dir.joinpath("model_{}.pkl".format(model_idx)), "wb") as f:
            pickle.dump(predictor_data, f)

    @staticmethod
    def _clean_up_after_fold(tmp_folder):
        logger.info("Start Ag clean up...")
        # - Delete tmp contents
        rmtree(tmp_folder / "models")
        rmtree(tmp_folder / "utils")

        for file in ["__version__", "learner.pkl", "metadata.json", "predictor.pkl"]:
            os.remove(tmp_folder / file)
        logger.info("Finished.")

    # -- Build Metatask Code
    def set_constraints(self, metric_name, time_limit):
        self.metatask.selection_constraints["manual"] = True
        self.metatask.selection_constraints["autogluon"] = dict(
            metric=metric_name,
            time_limit=time_limit,
            num_cpus="auto",
            presets="best_quality",
            calibrate=False,
            fit_weighted_ensemble=False
        )

    def _load_predictor_data_for_metatask(self, fold_idx, model_id, classes_=None):
        store_dir = self.tmp_output_dir.joinpath("fold_{}/.ag_assembler".format(fold_idx)).joinpath(
            "prediction_data")

        with open(store_dir.joinpath("model_{}.pkl".format(model_id)), "rb") as f:
            predictor_data = pickle.load(f)

        config = predictor_data["bm_config"]
        check_sum_for_name = hashlib.md5(str(config).encode('utf-8')).hexdigest()
        predictor_name = config["model_name"] + "({})".format(str(check_sum_for_name))
        predictor_description = {
            "autogluon-model": True,
            "config": config,
            "fit_time": float(predictor_data["fit_time"]),
            "predict_time": float(predictor_data["predict_time"]),
            "model_evaluated_time": int(predictor_data["model_evaluated_time"])
        }

        predictor_description["config"]["stack_level"] = int(predictor_description["config"]["stack_level"])
        predictor_description["config"]["score_val"] = float(predictor_description["config"]["score_val"])

        # Get Predictions
        if classes_ is not None:
            # Classification
            test_confs = predictor_data["test_y_pred"]
            test_y_pred = classes_.take(np.argmax(test_confs, axis=1), axis=0)
            val_confs = predictor_data["val_y_pred"]
            val_y_pred = classes_.take(np.argmax(val_confs, axis=1), axis=0)
            validation_data = [(fold_idx, val_y_pred, val_confs, predictor_data["val_indices"])]
        else:
            # Regression case
            raise NotImplementedError

        # --- Check Prediction Data
        if not np.isfinite(test_confs).all():
            raise ValueError("Test Confidences contain non-finite values.")

        if not np.isfinite(validation_data[0][2]).all():
            raise ValueError("Validation Confidences contain non-finite values.")

        return predictor_name, predictor_description, test_y_pred, test_confs, validation_data

    def evaluate_existing_predictors(self, metric):
        """ Get the score and other values needed to filter predictors.
        """
        eval_results = {f: [] for f in self.folds_to_run}
        # Iterate over all predictors and compute score
        for iter_idx, fold_idx in enumerate(self.folds_to_run, 1):
            logger.info("### Pre-Processing Fold {} | {}/{} ###".format(fold_idx, iter_idx, len(self.folds_to_run)))

            # --- Parse File Structure
            base_dir = self.tmp_output_dir.joinpath("fold_{}".format(fold_idx))
            assembler_dir = base_dir.joinpath(".ag_assembler")
            if not assembler_dir.exists():
                raise ValueError("No Predictor Data Exists!")

            pred_data_dir_names = [os.path.basename(run_folder) for run_folder in
                                   glob.glob(str(assembler_dir.joinpath("prediction_data/model_*.pkl")))]
            n_pred_data = len(pred_data_dir_names)

            if self.classification:
                classes_ = np.load(assembler_dir.joinpath("classes_.npy"), allow_pickle=True)
                le_ = LabelEncoder().fit(classes_)
            else:
                raise NotImplementedError("Regression not yet implemented!")

            # --- Iterate over base models
            for pred_d_idx, pred_d_identifier in enumerate(pred_data_dir_names, 1):
                # -- Handle IDs
                model_id = pred_d_identifier.split("_")[1].split(".")[0]
                logger.info("# Evaluating Prediction Data for Model {} | {}/{} #".format(model_id, pred_d_idx,
                                                                                         n_pred_data))

                # --- Load Prediction Data
                _, predictor_description, _, _, _ = self._load_predictor_data_for_metatask(fold_idx, model_id,
                                                                                           classes_=classes_)

                eval_time = predictor_description["model_evaluated_time"]
                filter_data = (predictor_description["config"]["score_val"], eval_time)

                # --- Save
                eval_results[fold_idx].append((model_id, filter_data))

        return eval_results

    def filter_predictors(self, fold_eval_data, pruner, min_n_predictor=2, top_n=50):
        """

        Parameters
        ----------
        fold_eval_data: (model__id, (filter_data))
            filter_data = (score,eval_time)
        """

        if len(fold_eval_data) < min_n_predictor:
            return []

        # Return immediately if less than top n models exit
        if len(fold_eval_data) <= top_n:
            return [ask_id for ask_id, _ in fold_eval_data]

        predictor_over_time = sorted(fold_eval_data, key=lambda x: x[-1][2])

        # -- Filter Predictors
        if pruner == "TopN":
            top_n_heap = []

            for model_id, (score, _, _) in predictor_over_time:
                _add_to_top_n(top_n_heap, top_n, (score, model_id))

            preds_to_keep = [ask_run_id for _, ask_run_id in top_n_heap]

        elif pruner == "SiloTopN":
            raise ValueError("SiloTopN is not supported for AutoGluon")
        else:
            raise ValueError("Unknown Pruner: {}".format(pruner))

        return preds_to_keep

    def build_metatask_from_predictor_data(self, pruner=None, metric=None):
        """Build a metatask from stored prediction data

        Parameters
        ----------
        pruner: str in ["TopN", "SiloTopN"] or None, default=None
            Filters the predictors before returning the metatask.
        metric: Callable
            Metric used to determine performance for filter predictors.

        Returns
        -------
        metatask
        """

        logger.info("Start Building Metatask from predictor data...")
        if not self.tmp_output_dir.exists():
            raise ValueError("No AutoGluon Output Data exists.")

        # ----- Determine which models to load into metatask
        if pruner is not None:
            # Input validation
            if pruner not in ["TopN", "SiloTopN"]:
                raise ValueError("filter_predictors is a wrong value. Got: {}".format(pruner))
            if metric is None:
                raise ValueError("Require a metric to filter predictors!")

            logger.info("Pre-Filter Predictors")

            eval_results = self.evaluate_existing_predictors(metric)

            to_load_predictors_per_fold = {f: self.filter_predictors(f_e_d, pruner)
                                           for f, f_e_d in eval_results.items()}

            # Report / Handle Removal Results
            for f, model_ids_to_keep in to_load_predictors_per_fold.items():
                n_all = len(eval_results[f])
                if not model_ids_to_keep:
                    # Not enough base models, break and stop here.
                    logger.info(
                        "No enough predictors exist. Fold: {}, N_Predictor: {}".format(f, n_all))
                    logger.info("Stopping Metatask Builder. No need to build a Metatask for this dataset.")
                    return None

                logger.info("Removed {} predictors due to Pruner Settings for fold {}".format(
                    n_all - len(model_ids_to_keep), f))
        else:
            raise ValueError("No pruner specified is not supported!")

        # ----- Given the IDs from before, load data and build metatask
        for iter_idx, (fold_idx, model_ids_to_load) in enumerate(to_load_predictors_per_fold.items(), 1):
            logger.info("### Processing Fold {} | {}/{} ###".format(fold_idx, iter_idx, len(self.folds_to_run)))

            # --- Parse File Structure
            assembler_dir = self.tmp_output_dir.joinpath("fold_{}/.ag_assembler".format(fold_idx))
            n_to_load = len(model_ids_to_load)

            if self.classification:
                classes_ = np.load(assembler_dir.joinpath("classes_.npy"), allow_pickle=True)
            else:
                classes_ = None

            # --- Iterate over base models
            for inner_iter_idx, model_id in enumerate(model_ids_to_load, 1):
                logger.info("# Processing Prediction Data for Model {} | {}/{} #".format(model_id, inner_iter_idx,
                                                                                         n_to_load))

                # --- Load Prediction Data
                predictor_name, predictor_description, test_y_pred, test_confs, validation_data \
                    = self._load_predictor_data_for_metatask(fold_idx, model_id, classes_=classes_)

                # -- Add data to metatask
                self.metatask.add_predictor(predictor_name, test_y_pred, confidences=test_confs,
                                            conf_class_labels=list(classes_),
                                            predictor_description=predictor_description,
                                            validation_data=validation_data,
                                            fold_predictor=True, fold_predictor_idx=fold_idx)

        logger.info("Finished Building Metatask from predictor data.")
        return self.metatask


def _add_to_top_n(heap, n, heap_item, key=lambda x: x[0]):
    if len(heap) < n:
        heappush(heap, heap_item)
    elif key(heap[0]) < key(heap_item):
        heappop(heap)
        heappush(heap, heap_item)
