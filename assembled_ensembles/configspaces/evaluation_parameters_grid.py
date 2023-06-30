def get_ens_weighting_config_space(seed_function_individual_fold):
    from ConfigSpace import ConfigurationSpace, Categorical, EqualsCondition, InCondition

    cs = ConfigurationSpace(
        name="evaluation_parameters_ensw_benchmark",
        seed=42,
        meta={"seed_function_individual_fold": seed_function_individual_fold, "rng_seed": 315185350},
    )

    # -- Basic Space
    hp_methods = Categorical("method", ["SingleBest", "LogisticRegressionStacking", "EnsembleSelection", "CMA-ES"])

    # -- EnsembleSelection Parameters
    hp_use_best = Categorical("use_best", [True])
    cond_use_best = EqualsCondition(hp_use_best, hp_methods, "EnsembleSelection")

    # -- CMA-ES Parameters
    hp_batch_size = Categorical("batch_size", ["dynamic"])
    cond_batch_size = EqualsCondition(hp_batch_size, hp_methods, "CMA-ES")

    hp_normalize_weights = Categorical("normalize_weights", ["no", "softmax"])
    cond_normalize_weights = EqualsCondition(hp_normalize_weights, hp_methods, "CMA-ES")

    hp_trim_weights = Categorical("trim_weights", ["no", "ges-like-raw", "ges-trim-round"])
    cond_trim_weights = InCondition(hp_trim_weights, hp_normalize_weights, ["softmax"])

    hp_single_best_fallback = Categorical("single_best_fallback", [False])
    cond_single_best_fallback = EqualsCondition(hp_single_best_fallback, hp_methods, "CMA-ES")

    hp_weight_vector_ensemble = Categorical("weight_vector_ensemble", [False])
    cond_weight_vector_ensemble = EqualsCondition(hp_weight_vector_ensemble, hp_methods, "CMA-ES")

    hp_bounded = Categorical("bounded", [False])
    cond_bounded = EqualsCondition(hp_bounded, hp_methods, "CMA-ES")

    hp_sigma0 = Categorical("sigma0", [False])
    cond_sigma0 = EqualsCondition(hp_sigma0, hp_methods, "CMA-ES")

    hp_x0 = Categorical("start_weight_vector_method", [False])
    cond_x0 = EqualsCondition(hp_x0, hp_methods, "CMA-ES")

    # -- Construct ConfigSpace
    cs.add_hyperparameters([
        hp_methods, hp_use_best, hp_batch_size, hp_normalize_weights, hp_single_best_fallback,
        hp_weight_vector_ensemble, hp_bounded, hp_trim_weights, hp_sigma0, hp_x0
    ])
    cs.add_conditions([
        cond_use_best, cond_batch_size, cond_normalize_weights, cond_single_best_fallback, cond_weight_vector_ensemble,
        cond_bounded, cond_trim_weights, cond_sigma0, cond_x0
    ])

    return cs


def get_config_space(return_grid=False):
    from ConfigSpace.util import generate_grid

    def seed_function_individual_fold(seed, fold_str):
        return seed + int(fold_str)

    cs = get_ens_weighting_config_space(seed_function_individual_fold)

    if return_grid:
        return cs, generate_grid(cs)

    return cs


def get_name_grid_mapping():
    import json
    import os
    from pathlib import Path

    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    with open(file_path / f"name_grid_mapping.json", "r") as f:
        name_grid_mapping = json.load(f)

    return name_grid_mapping


# -- utils
def _config_to_unique_name(config):
    ks = sorted(list(config.keys()))
    ks.remove("method")
    filtered_ks = [k for k in ks if ((not isinstance(config[k], bool)) or (config[k]))]
    string = str(config["method"])
    for k in filtered_ks:
        if isinstance(config[k], bool):
            string += f"|{k}"
        else:
            string += f"|{k}:{config[k]}"

    return string


def _create_name_grid_map(grid_cs):
    import json

    name_map = {_config_to_unique_name(conf): dict(conf) for conf in grid_cs}

    with open(f"name_grid_mapping.json", "w") as outfile:
        outfile.write(json.dumps(name_map))


def _run_local():
    out_cs, grid_cs = get_config_space(return_grid=True)
    print(out_cs)
    print(len(grid_cs))
    _create_name_grid_map(grid_cs)


if __name__ == "__main__":
    _run_local()
