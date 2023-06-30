import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "../.."))  # have to do this because of singularity. I hate it

from pathlib import Path
from assembled_autogluon.util.supported_metrics import msc
from assembled_autogluon.util.metatask_base import get_metatask
from assembled_autogluon.ag_assembler import AgAssembler

if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    time_limit = int(sys.argv[2])
    folds_to_run = [int(x) for x in sys.argv[3].split(",")] if "," in sys.argv[3] else [int(sys.argv[3])]
    metric_name = sys.argv[4]
    benchmark_name = sys.argv[5]

    # -- Build paths
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_output_dir = file_path.parent.parent / "benchmark/output/{}/task_{}".format(benchmark_name, openml_task_id)
    print("Full Path Used: {}".format(tmp_output_dir))

    # -- Get The Metatask
    print("Building Metatask for OpenML Task: {}".format(openml_task_id))
    mt = get_metatask(openml_task_id)

    if openml_task_id == "-1":
        time_limit = time_limit / 60

    metric_to_optimize = msc(metric_name, len(mt.class_labels) == 2)

    # -- Init and run assembler
    print("Run Assembler")
    assembler = AgAssembler(mt, tmp_output_dir, folds_to_run=folds_to_run)
    assembler.run(metric_to_optimize, time_limit)

    print("Finished Run, Save State")
    for fold in folds_to_run:
        s_path = file_path.parent.parent / "benchmark/state/{}/task_{}/".format(benchmark_name, openml_task_id)
        s_path.mkdir(parents=True, exist_ok=True)
        (s_path / "run_ag_on_metatask_{}.done".format(fold)).touch()
