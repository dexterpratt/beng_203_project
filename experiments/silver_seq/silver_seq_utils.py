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

DATA_DIR = Path(__file__).resolve().parent

def load_silver_seq_data():
    counts_path = DATA_DIR / 'silver_seq_counts.txt'
    silver_seq_counts = pd.read_csv(counts_path, sep="\t")
    silver_seq_counts = silver_seq_counts.astype(int)
    # Our data, X, is swapped, features are rows. we need to rotate the array.
    X = silver_seq_counts.T
    gene_mappings = pd.read_csv(DATA_DIR / 'gene_mappings.csv')

    silver_seq_metadata = pd.read_excel(DATA_DIR / 'silver_seq_metadata.xlsx')
    return X, silver_seq_counts, silver_seq_metadata, gene_mappings

def max_tpm_features(X, threshold):
    X = X.loc[:, X.max() > threshold]
    y = [row.split('_')[0] for row in X.index]
    return X, y

def variance_features(X, threshold):
    X = X.loc[:, X.var() > threshold]
    y = [row.split('_')[0] for row in X.index]
    return X, y

def silver_seq_classify(X, y, clf = None, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

def filter_by_gene_symbols(symbol_list, gene_mappings, X):

    return 
