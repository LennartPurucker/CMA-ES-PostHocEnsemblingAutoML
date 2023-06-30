# --- Factory
def cma_es_factory(rng_seed, metric, n_jobs, batch_size, normalize_weights, single_best_fallback,
                   weight_vector_ensemble,
                   bounded, trim_weights, sigma0, start_weight_vector_method):
    from assembled_ensembles.methods.numerical_solvers.cmaes import CMAES
    from numpy.random import RandomState

    return {
        "technique": CMAES,
        "technique_args": {
            "n_iterations": 50,
            "score_metric": metric,
            "batch_size": batch_size,
            "random_state": RandomState(rng_seed),
            "normalize_weights": normalize_weights,
            "single_best_fallback": single_best_fallback,
            "weight_vector_ensemble": weight_vector_ensemble,
            "start_weight_vector_method": start_weight_vector_method,
            "trim_weights": trim_weights,
            "bounded": bounded,
            "n_jobs": n_jobs,
            "sigma0": sigma0
        },
        "pre_fit_base_models": True
    }