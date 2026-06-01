import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold

from pathlib import Path
import ndex2
from ndex2.cx2 import CX2Network, RawCX2NetworkFactory
import json
import silver_seq_utils as utils

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
# -------------------------------------------------------

def rank_deg(X, group_a="AD", group_b="N"):
    a_cols = X.columns[X.columns.str.startswith(group_a + "_")]
    #print(a_cols)
    b_cols = X.columns[X.columns.str.startswith(group_b + "_")]
    #print(b_cols)
    A = X[a_cols].to_numpy()
    B = X[b_cols].to_numpy()

    mean_a = A.mean(axis=1)
    mean_b = B.mean(axis=1)
    # print(mean_b[0:4])
    log2fc = np.log2((mean_a + 1)/(mean_b + 1))
    # print(log2fc[0:4])

    pvals = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        try:
            _, p = mannwhitneyu(A[i], B[i], alternative="two-sided")
        except ValueError:
            p = 1.0
        pvals[i] = p

    _, qvals = fdrcorrection(pvals) # Benjamini-Hochberg
    print(qvals[0:4])

    result = pd.DataFrame({
        "ensemble": X.index,
        "mean_AD": mean_a,
        "mean_N": mean_b,
        "log2fc": log2fc,
        "pval": pvals,
        "qval": qvals
    }).set_index("ensemble")
    result = result.sort_values("qval")
    result.to_csv("deg_results.csv")


# -------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent

go_gene_sets = pd.read_csv(DATA_DIR / "data/gene_sets.csv")

X, silver_seq_counts, silver_seq_meta = utils.load_silver_seq_data()

print(f'genes before filtering = {X.shape}')

# print(X.head())

min_threshold = 100

max_threshold = 10000

# for i in range(5):
#     m = X.iloc[:,i].max()
#     print(f'max = {m}')

X = X.loc[X.max(axis=1) >= min_threshold]

#X = X.loc[X.max(axis=1) <= max_threshold]


print(f'genes after filter between {min_threshold} and {max_threshold} = {X.shape}')

result = rank_deg(X)

result = result.loc[result["pval"] <= 0.1]

result_translated = utils.translate_ensemble(result.index)

symbols = result_translated.dropna().tolist()

symbol_string = " ".join(symbols)

print(symbol_string)