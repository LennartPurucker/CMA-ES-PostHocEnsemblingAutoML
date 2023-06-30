# --- Factories
def _factory_es(rng_seed, metric, n_jobs, use_best):
    from assembled_ensembles.methods.ensemble_selection.greedy_ensemble_selection import EnsembleSelection
    from numpy.random import RandomState

    return {
        "technique": EnsembleSelection,
        "technique_args": {"n_iterations": 50,
                           "metric": metric,
                           "n_jobs": n_jobs,
                           "random_state": RandomState(rng_seed),
                           "use_best": use_best},
        "pre_fit_base_models": True
    }