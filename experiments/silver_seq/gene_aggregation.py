'''
Aggregation utils

Gene set to feature:
- select gene rows to df
- score df to column
- put on feature set
- input: df1 where the rows are genes
- for each gene set, select the df1 rows matching the ensemble ids
    - make a row with the calculated values for each column
- the output is df2 where the rows are aggregation features
- and the columns are samples

Adapt Laura's metadata code
- get df of selected metadata

Aggregation by subject
- average value
- variance

Merge all features to df

Run multiple variants of methods/features in batches, produce comparison reports

Feature creation

- Mitochondrial Genes
- Networks based on relevant processes, select from filtered genes
    - autophagy
    - neurodegeneration
    - alzheimer's
    - inflammation types
    - ox-phos
    - mtor
    - cell cycle
    - hypoxia
    - metabolic
    - cell death
        - caspase
    - rna binding
    - cytosolic rna sensing
    - growth and proliferation

GO biological processes filtered by term size? And overlap?

Poorly characterized genes:
how many genes are poorly characterized AND are highly differentially expressed or highly variable?
scoop them up by gene families?

- Diffusion based expansion of gene sets to add unclassified genes
- Correlation clusters


Gene filters
- Priyan: To reduce noise and minimize overfitting, low-expression genes were filtered prior to analysis, followed by CPM normalization and log2 transformation. 
- differential expression in training set

Try experiments with cross-validation, learn how that works

'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import ndex2
from ndex2.cx2 import CX2Network, RawCX2NetworkFactory
import json
import silver_seq_utils as utils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).resolve().parent

top_variable = [
    "MTND2P28", "MTCO1P12", "MTATP6P1", "MACF1", "TTN", "PPBP", "MTND6P4",
    "MTND4P12", "ACTB", "TMSB4X", "KCNQ1OT1", "HBB", "NAP1L1", "HELLPAR",
    "RN7SL2", "B2M", "RORA", "MT-TF", "MT-RNR1", "MT-TV", "MT-RNR2", "MT-ND1",
    "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3", "MT-ND3",
    "MT-ND4L", "MT-ND4", "MT-TH", "MT-ND5", "MT-ND6", "MT-CYB", "MT-TP",
]

def get_ensemble(symbols, mappings=None):
    symbol_list=list(symbols)
    return list(mappings.loc[mappings["symbol"].isin(symbol_list), "query"])

def create_go_gene_sets():
    mappings = pd.read_csv(DATA_DIR / "gene_mappings.csv", delimiter=',')
    term_symbol_path = DATA_DIR / 'data/collapsed_go.symbol'
    term_symbol = pd.read_csv(term_symbol_path, sep="\t", names=["term", "symbol", "type"], header=None)
    term_symbol = term_symbol.loc[term_symbol["type"] != "default"]
    # print(term_symbol.shape)
    term_name_path = DATA_DIR / 'data/goID_2_name.tab'
    term_name = pd.read_csv(term_name_path, sep="\t", names=["term", "name"], header=None)
    term_name = term_name.set_index("term")["name"] 
    # print(term_name.shape)
    # print(term_name.head(10))
    gene_sets = term_symbol.groupby('term')['symbol'].apply(list).reset_index()
    gene_sets.columns = ["term", "symbols"]
    gene_sets["count"] = gene_sets["symbols"].str.len()
    gene_sets["name"] = gene_sets["term"].map(term_name)

    # gene_sets["name"] = term_symbol.loc[term_symbol["term"]]["name"]
    gene_sets = gene_sets.loc[gene_sets["count"].isin(range(3, 50))]
    # test = gene_sets.head(10)
    # foo = get_ensemble(top_variable, mappings=mappings)
    # print(f'test translate: {foo}')
    # test["ensemble"] = test["symbols"].apply(get_ensemble, mappings=mappings)
    # ensemble_ids = gene_sets["symbols"].apply(get_ensemble, mappings=mappings)[0]
    # gene_sets["ensemble"] = " ".join(ensemble_ids)
    gene_sets["ensemble"] = gene_sets["symbols"].apply(
        lambda syms: " ".join(get_ensemble(syms, mappings=mappings))
    )

  # print(gene_sets["count"].max())
    # print(gene_sets.shape)
    # print(gene_sets.head(10))
    # print(test)
    gene_sets.to_csv('experiments/silver_seq/data/gene_sets.csv')
    return gene_sets

# create_go_gene_sets()

# def feature_row(X, ids):
#     fx = X.loc[X.index.isin(ids)]
#     return fx.mean(numeric_only=True).to_frame().T

def gene_set_features(X, gene_sets, min_genes, max_genes):
    # input: X where the rows are genes
    rows = []
    terms = []
    # for each gene set, select the X rows matching the ensemble ids
    for i in gene_sets.index:
        idstring = gene_sets.loc[i, "ensemble"]
        ids = idstring.split(" ")
        filtered_genes = X.loc[X.index.isin(ids)]
        n_matched = len(filtered_genes)
        if n_matched >= min_genes and n_matched <= max_genes:
            row = filtered_genes.var(axis=0, numeric_only=True).to_frame().T
            terms.append(gene_sets.loc[i, "name"])
            rows.append(row)
    if len(terms) == 0:
        return None
    features = pd.concat(rows)
    features.index = terms
    return features

def filter_by_tpm(X, threshold):
    X = X.loc[:, X.max() > threshold]
    y = [row.split('_')[0] for row in X.index]
    return X, y

#------------------------------------------------------

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

X = X.loc[X.max(axis=1) <= max_threshold]


print(f'genes after filter between {min_threshold} and {max_threshold} = {X.shape}')

min_genes = 1
max_genes = 10

X_gs = gene_set_features(X, go_gene_sets, min_genes, max_genes)

'''
- Networks based on relevant processes, select from filtered genes
    - autophagy
    - neurodegeneration
    - alzheimer's
    - inflammation types
    - ox-phos
    - mtor
    - cell cycle
    - hypoxia
    - metabolic
    - cell death
        - caspase
    - rna binding
    - cytosolic rna sensing
    - growth and proliferation
'''

filter_string = "autophag"

X_gs = X_gs.loc[X_gs.index.str.contains(filter_string)]

print(f'gene sets {min_genes} to {max_genes} and name contains {filter_string} = {X_gs.shape}')

meta_cols = ['age_at_collection', 'sex', 'year_sample', 'apoe']

meta = silver_seq_meta.set_index("sample_id_alias")[meta_cols]

meta = pd.get_dummies(meta[meta_cols], drop_first=True)

meta_features = meta.reindex(X_gs.columns).T

X_all = pd.concat([X_gs, meta_features])

# transpose so samples are rows and features are columns (sklearn convention)
X_all = X_all.T

y = [s.split('_')[0] for s in X_all.index]

# print("y:")
# print(y)

# model_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('logreg', LogisticRegression(max_iter=5000, random_state=42)) # Keep increased max_iter
# ])

# utils.silver_seq_classify(X2, y, model_pipeline)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
print(X_train.head())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'X_train shape = {X_train.shape}')
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
utils.plot_roc(clf, X_test, y_test)
    

# --- Plot 2: Feature Importance ---
# We extract the importance scores and map them to column names
importances = clf.feature_importances_
feature_names = X_all.columns
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=True)
forest_importances = forest_importances[-10:]

plt.figure(figsize=(10, 6))
forest_importances.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score (Gini Importance)')
plt.ylabel('Features')
plt.tight_layout()
plt.show()