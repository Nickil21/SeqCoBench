import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score

from source.config import LLM_BASED, MATCH_BASED, METRIC_COLUMNS, SEMANTIC_PRESERVING, SEMANTIC_ALTERING, METRIC_MAPPING, LLM_PROMPT_BASED

from source.utils.latex import tables

# test_metrics = pd.read_csv("/bask/projects/j/jlxi8926-auto-sum/nmaveli/datasets/python/ours/test_metrics.csv")

test_metrics = pd.read_csv("/bask/projects/j/jlxi8926-auto-sum/nmaveli/datasets/python/ours/test_metrics_cot.csv")

metric_ap_result = []
for metric in LLM_PROMPT_BASED: #METRIC_COLUMNS:
    print(metric)
    out = {}
    metric_ap = average_precision_score(test_metrics['label'], test_metrics[metric]) * 100
    out['Metric'] = metric
    out['AP'] = round(metric_ap, 2)
    metric_ap_result.append(out)

df1 = pd.DataFrame(metric_ap_result)
df1["Type"] = np.where(df1["Metric"].isin(MATCH_BASED), r"\rotatebox{90}{Match-based}", r"\rotatebox{90}{LLM-based}")
df1["Metric"] = df1["Metric"].map(METRIC_MAPPING)
df1.set_index(["Type", "Metric"], inplace=True)

print(df1)

df1.style.format(precision=2).apply(tables.highlight_max).to_latex(buf=f"source/outputs/results/latex/ours_ap_breakdown_two_shot.tex",
                        column_format="llr",
                        position="htbp",
                        position_float="centering",
                        hrules=True,
                        clines="skip-last;data",
                        multicol_align="c",
                        
                        label="table:mbpp-derived-results",
                        caption=f"Results on the MBPP-derived test set in the two-shot setting using the average precision metric."
                    )
