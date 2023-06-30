import pandas as pd
import json

bacc_df = pd.read_csv("../evaluation/tmp_results/bacc_metatask_analysis_results.csv")
roc_df = pd.read_csv("../evaluation/tmp_results/roc_auc_metatask_analysis_results.csv")

with open("../conf/task_data.json") as f:
    task_json = json.load(f)

with open("../conf/roc_auc_benchmark_data.json") as f:
    bm_json = json.load(f)

task_ids = bacc_df["TaskID"].unique()

cols = ["Dataset Name", "OpenML Task ID", "#instances", "#features", "#classes", "Memory (GB)",
        "B - avg. # base models",
        "R - avg. # base models",
        "B - avg. # distinct algorithms",
        "R - avg. # distinct algorithms"
        ]
rows = []
for task_id in task_ids:
    row = [task_json[str(task_id)]["dataset_name"], task_id,
           task_json[str(task_id)]["n_instances"], task_json[str(task_id)]["n_features"],
           task_json[str(task_id)]["n_classes"]]

    # Same for roc and bacc
    row.append(bm_json["framework_memory_gbs"]
               + bm_json["special_cases"]["framework_memory_gbs"].get(str(task_id), 0))

    tmp_b = bacc_df[(bacc_df["TaskID"] == task_id)][["mean_base_models_count", "mean_distinct_algorithms_count"]]
    tmp_r = roc_df[(roc_df["TaskID"] == task_id)][ ["mean_base_models_count", "mean_distinct_algorithms_count"]]

    row.extend([
        tmp_b.iloc[0][0], tmp_r.iloc[0][0],
        tmp_b.iloc[0][1], tmp_r.iloc[0][1]
    ])
    rows.append(row)

res = pd.DataFrame(rows, columns=cols)
res.to_csv("../evaluation/out/data_overview.csv", index=False)
