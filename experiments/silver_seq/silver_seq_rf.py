from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import silver_seq_utils as utils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

'''
counts_path = '/Users/idekeradmin/Dropbox/GitHub/AD_prediction_blood/experiments/silver_seq/silver_seq_counts.txt'
silver_seq_counts = pd.read_csv(counts_path,  sep="\t")
silver_seq_counts = silver_seq_counts.astype(int)

silver_seq_metadata = pd.read_excel('/Users/idekeradmin/Dropbox/GitHub/AD_prediction_blood/experiments/silver_seq/silver_seq_metadata.xlsx')

# for a given train and test, the data needs to be in a
# X : a 2D array of samples x features - samples are rows, features are columns
# y : a 1D array of labels
# y values can be integers, string, other - sklearn encodes them internally


# the split between training and test is accomplished by sklearn train_test_split
# which 

# If we include metadata features such "sex" with n non-numeric categorical values, e.g., "F, M" 
# those require encoding. Sex, in this dataset, is binary, so we just map to 1, 0.
# For categories with multiple values, such as APOE status, the sklearn option is "one hot" 
# which maps them to n binary inputs,

# Our data, X, is swapped, features are rows. we need to rotate the array.

X = silver_seq_counts.T

# We also have a large number of features. We need to filter/aggregate the features

# test: features = genes with the highest maximum TPM
# pandas methods:
# value based on column values X.<op>(). sum, max, etc.
# filter columns by slicing: X.loc[:, val <comparison> threshold]

def max_tpm_features(X, threshold):
    X = X.loc[:, X.max() > threshold]
    y = [row.split('_')[0] for row in X.index]
    return X, y

def variance_features(X, threshold):
    X = X.loc[:, X.var() > threshold]
    y = [row.split('_')[0] for row in X.index]
    return X, y

def silver_seq_classify(X, y, clf = RandomForestClassifier()):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'X_train shape = {X_train.shape}')
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
    plot_roc(clf, X_test, y_test)

def plot_roc(clf, X_test, y_test):
    # probability scores rather than classification
    # they are probabilities of positive class
    # AD is 0 based on default alphabetical sorting of classes
    y_scores = clf.predict_proba(X_test)[:, 0] 
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label='AD')
    roc_auc = auc(fpr, tpr)
    # standard ROC plot:
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.show()
'''

X, silver_seq_counts, silver_seq_counts = utils.load_silver_seq_data()

# X, y = max_tpm_features(X, 500)

X, y = utils.variance_features(X, 4000)

print(X)

print(X.columns)

symbols = [str(x) for x in utils.translate_ensemble(list(X.columns))]
symbol_string = " ".join(symbols)
print(symbol_string)

# cx2_network = utils.get_ndex_network_by_id("49e43d68-939b-11ea-aaef-0ac135e8bacf")

# symbol_list = utils.get_node_names(cx2_network, name_field="livia name")

cfrna_ad_candidates = [
    # Neuron-enriched, abundant, brain-origin signal
    "SNAP25",      # presynaptic, neuron-specific, abundant
    "SYT1",        # synaptotagmin-1, neuronal
    "NEFL",        # neurofilament light — protein already validated as blood biomarker
    "NEFM",
    "GAP43",
    "STMN2",       # neuron-specific, lost early in neurodegeneration
    "RBFOX3",      # NeuN, pan-neuronal
    "MAP2",
    "ENO2",        # neuron-specific enolase
    "MAPT",        # tau — AD-relevant, neuronal

    # Entorhinal/hippocampal excitatory neuron markers (the dying population)
    "RELN",        # reelin, layer II entorhinal — selectively vulnerable
    "CALB1",
    "CARTPT",

    # Death-pathway markers (less brain-specific, but mechanistically informative)
    "MLKL",
    "RIPK3",
    "ZBP1",
    "ACSL4",
    "CHAC1",

    # Brain-enriched miRNAs — better cfRNA stability, EV-packaged
    "MIR9",        # miR-9, neuron-enriched
    "MIR124",      # miR-124, most abundant brain miRNA
    "MIR132",      # synaptic, decreased in AD
    "MIR128",
    "MIR219A1",

    # Long non-coding, brain-enriched
    "BCYRN1",      # BC200, neuron-specific, dendritic
    "MALAT1",      # abundant, though not brain-specific
]

# print(f'symbols = {cfrna_ad_candidates}')

# X = utils.filter_genes_by_symbols(X, cfrna_ad_candidates, exclude=False)

# y = [row.split('_')[0] for row in X.index]

# print(X.shape)

# model_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('logreg', LogisticRegression(max_iter=5000, random_state=42)) # Keep increased max_iter
# ])

# utils.silver_seq_classify(X, y, model_pipeline)

#utils.silver_seq_classify(X, y, RandomForestClassifier(n_estimators=500))

# silver_seq_classify(X, y, MultinomialNB())

