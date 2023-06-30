import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "../.."))  # have to do this because of singularity. I hate it

from pathlib import Path
from shutil import rmtree
from functools import partial

from assembled_autogluon.util.supported_metrics import msc
from assembled_autogluon.util.metatask_base import get_metatask
from assembled_autogluon.ag_assembler import AgAssembler

if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    time_limit = int(sys.argv[2])
    metric_name = sys.argv[3]
    file_format = sys.argv[4]
    benchmark_name = sys.argv[5]
    # -- Default to TopN as it makes no difference
    #   We set this to prune to top 50 but AutoGluon never produced more than 50 models.
    filter_predictors = sys.argv[6] if len(sys.argv) > 6 else "TopN"

    if file_format not in ["csv", "hdf", "feather"]:
        raise ValueError("Unknown file format: {}".format(file_format))

    # -- Build paths
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_output_dir = file_path.parent.parent / "benchmark/output/{}/task_{}".format(benchmark_name, openml_task_id)
    print("Full Path Used: {}".format(tmp_output_dir))

    # -- Clean Previous results if they exist
    out_path = tmp_output_dir.joinpath("final_output/{}".format(filter_predictors))
    print("Clean Up Previous Results")
    if out_path.exists():
        rmtree(out_path)
    os.makedirs(out_path)

    # -- Rebuild The Metatask
    print("Building Metatask for OpenML ID: {}".format(openml_task_id))
    mt = get_metatask(openml_task_id)
    mt.file_format = file_format
    mt.save_chunk_size = 500

    # -- Init and run assembler
    print("Run Assembler")
    assembler = AgAssembler(mt, tmp_output_dir)
    metric_to_optimize = msc(metric_name, len(mt.class_labels) == 2)

    if metric_to_optimize.name == "roc_auc_ovo_macro":
        partial(metric_to_optimize, labels=list(range(mt.n_classes)))

    assembler.set_constraints(metric_to_optimize.name, time_limit)
    mt = assembler.build_metatask_from_predictor_data(pruner=filter_predictors,
                                                      metric=metric_to_optimize)

    if mt is not None:
        print("Save To File")
        mt.to_files(out_path)
        print("Finished Saving to File")

    print("Save State")
    s_path = file_path.parent.parent / "benchmark/state/{}/task_{}/".format(benchmark_name, openml_task_id)
    s_path.mkdir(parents=True, exist_ok=True)
    (s_path / "run_full_metatask_builder_{}.done".format(filter_predictors)).touch()
