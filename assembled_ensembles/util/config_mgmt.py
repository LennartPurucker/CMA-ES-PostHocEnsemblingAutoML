from assembled_ensembles.default_configurations import ens_ensemble_selection, ens_other, ens_sklearn, \
    ens_solver


def get_all_ensemble_config_names():
    from assembled_ensembles.configspaces.evaluation_parameters_grid import get_name_grid_mapping

    return list(get_name_grid_mapping().keys())


def get_ensemble_switch_case_config(config, rng_seed=None, metric=None, n_jobs=None,
                                    is_binary=None, labels=None):
    method = config["method"]

    if method == "SingleBest":
        return ens_other.customSingleBest(metric=metric)
    elif method == "EnsembleSelection":
        return ens_ensemble_selection._factory_es(rng_seed, metric, n_jobs, config["use_best"])
    elif method == "CMA-ES":
        return ens_solver.cma_es_factory(rng_seed, metric, n_jobs, config["batch_size"], config["normalize_weights"],
                                         config["single_best_fallback"], config["weight_vector_ensemble"],
                                         config["bounded"], config["trim_weights"], config["sigma0"],
                                         config["start_weight_vector_method"])
    elif method == "LogisticRegressionStacking":
        return ens_sklearn.sklearnStackingClassifier(rng_seed=rng_seed, n_jobs=n_jobs)
    else:
        raise ValueError(f"Unknown method! Got: {method}")
