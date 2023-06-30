# -- Sklearn
def sklearnStackingClassifier(rng_seed=None, n_jobs=-1, **kwargs):
    # "sklearn.StackingClassifier"
    from assembled_ensembles.methods.sklearn.stacking import StackingClassifier
    from assembled_ensembles.methods.sklearn.custom_logistic_regression import LogisticRegression
    from numpy.random import RandomState
    from assembled_ensembles.util.preprocessing import get_default_preprocessing

    return {
        "technique": StackingClassifier,
        "technique_args": {"final_estimator": LogisticRegression(random_state=RandomState(rng_seed),
                                                                 solver="lbfgs",
                                                                 # Larger number such that this never triggers
                                                                 max_iter=1000000,
                                                                 # Max_fun set by stacking later!
                                                                 max_fun=0,
                                                                 n_jobs=n_jobs, verbose=1),
                           "prefitted": True, "only_fit_final_estimator": True, "max_fun_per_base_model": 50
                           },
        "pre_fit_base_models": True, "base_models_with_names": True, "label_encoder": True,
        "preprocessor": get_default_preprocessing()
    }