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

# # X, y = max_tpm_features(X, 500)

# X, y = utils.variance_features(X, 4000)

# symbols = [str(x) for x in utils.translate_ensemble(list(X.columns))]

# X = utils.filter_genes_by_symbols(X, symbols, exclude=True)

# y = [row.split('_')[0] for row in X.index]

# model_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('logreg', LogisticRegression(max_iter=1000, random_state=42)) # Keep increased max_iter
# ])

# utils.silver_seq_classify(X, y, model_pipeline)

X, y = utils.variance_features(X, 50)

# print(X)

# print(X.columns)

# symbols = [str(x) for x in utils.translate_ensemble(list(X.columns))]
# symbol_string = " ".join(symbols)
# print(symbol_string)

# cx2_network = utils.get_ndex_network_by_id("49e43d68-939b-11ea-aaef-0ac135e8bacf")

# symbol_list = utils.get_node_names(cx2_network, name_field="livia name")
neural_genes = [
    "AP2A1", "AP3D1", "APBA1", "APBA2", "ARF1", "ARF6", "ADD2", "ARFGEF1",
    "ARFGEF2", "ARFIP2", "ATP6V0C", "BAIAP3", "BET1", "BIN1", "BLOC1S1", "BSN",
    "CACNA1A", "CADPS1", "CADPS2", "CALM2", "CASK", "CLTC", "CNO", "CNTNAP",
    "CPLX2", "DNAJC5", "DNM1", "DOC2A", "DOC2B", "RPH3A", "EHD1", "EPN1",
    "EPS15", "ERC1", "ERC2", "EXOC6", "EXPH5", "FBXO45", "GAP43", "GDI2",
    "GOPC", "GOSR2", "HGS", "ICA1", "ITSN2", "KIF1A", "KIF1B", "LAMP1",
    "PTPRF", "LIN7A", "LPHN1", "MADD", "MSS4", "MUTED", "MYRIP", "NRXN1",
    "NSF", "PACSIN1", "PCLO", "PICALM", "PIK4CA", "PIP5K1C", "PLDN", "PPFIA1",
    "PPFIA2", "PPFIA3", "PPFIA4", "PSCD1", "PSCD2", "RAB3A", "RAB3B", "RAB3C",
    "RAB3D", "RAB3GAP", "RAB3IL1", "RAB6IP1", "RAB27A", "RAB27B", "RABAC1",
    "RABGAP1", "RALA", "RAPGEF4", "RGS7", "RGS9", "RGS11", "RILP", "RIMBP2",
    "RIMS1", "RIMS2", "SCAMP1", "SCG5", "SCIN", "SEC22B", "SH3GL2", "SIPA1L",
    "SLC17A7", "SNAP25", "SNAP29", "SNAPA", "SNAPIN", "SNIP", "SNPH", "SNX8",
    "SNX9", "SNX18", "SNX33", "STON1", "STON2", "STX1A", "STX1B", "STXBP1",
    "STXBP2", "STXBP3", "STXBP5", "STXBP5L", "STXBP6", "SV2A", "SYNGR1",
    "SYNGR2", "SYNGR3", "SYN1", "SYN2", "SYN3", "SYNJ1", "SYNJ2", "SYNPR",
    "SYNTABULIN", "SYP", "SYT1", "SYT2", "SYT3", "SYT4", "SYT7", "SYT8",
    "SYT9", "SYT11", "SYTL4", "SYTL5", "TMEM163", "TRAPPC1", "TSPAN12",
    "TXLNA", "UNC13A", "UNC13B", "UNC13C", "VAMP1", "VAMP2", "VAMP3", "VAMP7",
    "VAPA", "VAT1", "VPS18", "VPS33B", "VTI1B", "YWHAQ", "AKAP5", "ANKS1B",
    "ARC", "BAIAP2", "CACNG2", "CACNG3", "CACNG4", "CACNG5", "CACNG7",
    "CACNG8", "CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G", "CDH2", "CRIPT", "CLMN",
    "CTNNB1", "DLG1", "DLG2", "DLG3", "DLGAP1", "DLGAP2", "DLGAP3", "DLGAP4",
    "DNM2", "FEZ1", "FEZ2", "GRASP", "GRIP1", "GRIP2", "HOMER1", "HOMER2",
    "HOMER3", "HPCAL1", "ITPR1", "KALRN", "KCNJ2", "LRP4", "LRRC7", "MLLT4",
    "NIR2", "NLGN1", "NLGN2", "NLGN3", "PICK1", "PPP1R9A", "PPP1R9B", "PRKCC",
    "RAPSN", "RYR1", "RYR2", "RYR3", "SCN11A", "SCN2A", "SEPT5", "SEPT7",
    "SHANK1", "SHANK2", "SHANK3", "SYNGAP", "TRAPPC4", "VSNL1", "BRSK1",
    "BRSK2", "DLG4", "FREQ", "RASGRP1", "RASGRP2", "RASGRP3", "RASGRP4",
    "SCRIB", "SH3GL1", "SH3GL3", "ABLIM1", "ABLIM2", "ABLIM3", "EPB4", "ENAH",
    "EVL", "VASP", "MTAP2", "NUP62", "NUP62CL", "RUSC2", "STMN2", "TUBB3",
    "CRMP1", "CRMP2", "CRMP3", "CRMP4", "CRMP5", "CDK5", "AEBP1", "CPE",
    "CPN1", "CPXM1", "CPXM2", "CPZ", "ELAV1", "ELAV2", "ELAV3", "ELAV4",
    "FOX3", "GNB5", "LDB1", "LDB2", "NF1", "NPHS2", "STOM", "PCSK2", "PLCB1",
    "PLCB2", "PLCB3", "PLCB4", "RIC3", "UCHL1", "UNC119A", "UNC119B",
]

top_variable = [
    "MTND2P28", "MTCO1P12", "MTATP6P1", "MACF1", "TTN", "PPBP", "MTND6P4",
    "MTND4P12", "ACTB", "TMSB4X", "KCNQ1OT1", "HBB", "NAP1L1", "HELLPAR",
    "RN7SL2", "B2M", "RORA", "MT-TF", "MT-RNR1", "MT-TV", "MT-RNR2", "MT-ND1",
    "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3", "MT-ND3",
    "MT-ND4L", "MT-ND4", "MT-TH", "MT-ND5", "MT-ND6", "MT-CYB", "MT-TP",
]

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

#print(f'symbols = {neural_genes}')

X = utils.filter_genes_by_symbols(X, neural_genes + top_variable, exclude=False)

y = [row.split('_')[0] for row in X.index]

print(X.shape)

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=5000, random_state=42)) # Keep increased max_iter
])

utils.silver_seq_classify(X, y, model_pipeline)

# utils.silver_seq_classify(X, y, RandomForestClassifier(n_estimators=500))

# silver_seq_classify(X, y, MultinomialNB())

