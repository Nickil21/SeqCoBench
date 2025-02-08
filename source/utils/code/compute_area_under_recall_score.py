import re
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, auc

from source.utils.latex.tables import transformation_wise_breakdown
from source.utils.latex.tables import highlight_max

from source.utils.code.evaluation_metric import area_under_recall_curve
from source.config import LLM_BASED, MATCH_BASED, METRIC_COLUMNS, SEMANTIC_PRESERVING, SEMANTIC_ALTERING, METRIC_MAPPING, TRANSFORMATION_CATEGORIES

test_metrics = pd.read_csv("/bask/projects/j/jlxi8926-auto-sum/nmaveli/datasets/python/ours/test_metrics.csv")
print(test_metrics.shape)

print(test_metrics['transformation'].value_counts())

# for-loop check in the original code
test_metrics.loc[(~(test_metrics["original_code"].str.contains('|'.join(['for', 'while']))) & (test_metrics['transformation'] == "transformation_for_while_loop")), "label"] = np.nan
test_metrics.loc[(~(test_metrics["original_code"].str.contains('|'.join(['<', '>', '<=', '>=', '==', '!=']))) & (test_metrics['transformation'] == "transformation_operand_swap")), "label"] = np.nan

test_metrics.dropna(inplace=True)

test_metrics['label'] = test_metrics['label'].astype(int)

print(test_metrics['transformation'].value_counts())
# print(test_metrics[test_metrics['transformation'] == 'transformation_dead_code_insert'])

# test_metrics = pd.concat([test_metrics[test_metrics['label'] == 0], test_metrics[test_metrics['label'] == 1].head(200)])
print(test_metrics.columns.tolist())

print(test_metrics['label'].value_counts())

# test_metrics['chrf'] = test_metrics['chrf'].div(100)
test_metrics.loc[test_metrics['transformation'].str.contains("dissimilar"), "transformation"] = "dissimilar_code_injection"

TRANSFORMATIONS = SEMANTIC_PRESERVING + SEMANTIC_ALTERING


transformation_metric_result = []
for transformation in TRANSFORMATIONS:
    # print(transformation)
    if transformation in SEMANTIC_PRESERVING:
        pos_label = 1
    if transformation in SEMANTIC_ALTERING:
         pos_label = 0
    df = test_metrics.query("transformation == @transformation")
    for metric in METRIC_COLUMNS:
        # print(metric)
        out = {}
        out["Transformation"] = transformation
        transformation_metric = area_under_recall_curve(df['label'], df[metric], pos_label=pos_label)
        out["Metric"] = metric
        out['AURecallCurveScore'] = transformation_metric
        transformation_metric_result.append(out)



def add_transformation_category(row):
    for transformation_category, transformation in TRANSFORMATION_CATEGORIES.items():
        if row in transformation:
            return transformation_category


# print(test_metrics.query("transformation == 'transformation_add_sub_variable'")['label'].value_counts())
# print(test_metrics.query("transformation == 'transformation_add_sub_variable'")['rouge1'].value_counts())

out = pd.DataFrame(transformation_metric_result)

out.loc[out['Metric'].isin(MATCH_BASED), 'Eval Type'] = "Match-based"
out.loc[out['Metric'].isin(LLM_BASED), 'Eval Type'] = "LLM-based"

out[r'\makecell{Transformation \\ Category}'] = out['Transformation'].map(lambda x: add_transformation_category(x))

out.loc[out['Transformation'].isin(SEMANTIC_PRESERVING), 'Type'] = "SP"
out.loc[out['Transformation'].isin(SEMANTIC_ALTERING), 'Type'] = "SA"

out["Transformation"] = out["Transformation"].str.replace("transformation_", "").str.split("_").map(lambda x: [s.title() for s in x]).str.join(" ")
out["Metric"] = out["Metric"].map(METRIC_MAPPING)

out = out.set_index(["Type", r'\makecell{Transformation \\ Category}', "Transformation", "Eval Type", "Metric"]).unstack([3, 4]).sort_index(level=[0, 1, 2])

# print(out['AURecallCurveScore', 'Match-based'])
# print(out['AURecallCurveScore', 'LLM-based'])

transformation_wise_breakdown(data=out['AURecallCurveScore', 'Match-based'], 
                              path='source/outputs/results/latex/ours_transformation_breakdown_aurcscore_match_based.tex',
                              eval_type='Match-based',
                              metric='Area under the Recall curve metric',
                              subset=out['AURecallCurveScore', 'Match-based'].select_dtypes(include=np.number).columns.tolist())


transformation_wise_breakdown(data=out['AURecallCurveScore', 'LLM-based'], 
                              path='source/outputs/results/latex/ours_transformation_breakdown_aurcscore_llm_based.tex',
                              eval_type='LLM-based',
                              metric='Area under the Recall curve metric',
                              subset=out['AURecallCurveScore', 'LLM-based'].select_dtypes(include=np.number).columns.tolist())
