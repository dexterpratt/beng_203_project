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

DATA_DIR = Path(__file__).resolve().parent

#def load_silver_seq_data():
#    counts_path = DATA_DIR / 'silver_seq_counts.txt'
#    silver_seq_counts = pd.read_csv(counts_path, sep="\t")
#    silver_seq_counts = silver_seq_counts.astype(int)
#    # Our data, X, is swapped, features are rows. we need to rotate the array.
#    X = silver_seq_counts.T
#    # gene_mappings = pd.read_csv(DATA_DIR / 'gene_mappings.csv')
#
#    silver_seq_metadata = pd.read_excel(DATA_DIR / 'silver_seq_metadata.xlsx')
#    return X, silver_seq_counts, silver_seq_metadata

#includes additional metadata
def load_silver_seq_data():
    counts_path = DATA_DIR / 'silver_seq_counts.txt'
    #counts_path = os.path.join(REPO_PATH, 'experiments/silver_seq/silver_seq_counts.txt') #internal testing
    silver_seq_counts = pd.read_csv(counts_path, sep="\t")
    silver_seq_counts = silver_seq_counts.astype(int)
    # Our data, X, is swapped, features are rows. we need to rotate the array.
    X = silver_seq_counts.T
    # gene_mappings = pd.read_csv(DATA_DIR / 'gene_mappings.csv')

    silver_seq_metadata = pd.read_excel(DATA_DIR / 'silver_seq_metadata.xlsx')
    #silver_seq_metadata = pd.read_csv(os.path.join(REPO_PATH, 'experiments/silver_seq/silver_seq_metadata.csv')) #internal testing

    #load additional metadata
    supp_df = pd.read_csv(DATA_DIR / 'additionalSupplementalInfo.txt', sep='\t')
    #supp_df = pd.read_csv(os.path.join(REPO_PATH, 'experiments/silver_seq/additionalSupplementalInfo.txt'), sep='\t') #internal testing
    #avoid repeated columns in merge
    cols_to_keep = [col for col in supp_df.columns if col not in ['braak_stage', 'group']]
    supp_df_filtered = supp_df[cols_to_keep]
    #merge info
    combined_df = silver_seq_metadata.merge(supp_df_filtered, on='donor_id_alias', how='left')

    #find the max collection year for each donor, assume death age corresponds to last year sample collected (best approximation)
    combined_df['max_year'] = combined_df.groupby('donor_id_alias')['year_sample'].transform('max')
    combined_df['age_at_collection'] = combined_df['age_at_death'] - (combined_df['max_year'] - combined_df['year_sample'])

    silver_seq_metadata = combined_df.copy()

    return X, silver_seq_counts, silver_seq_metadata

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

def translate_symbols(symbol_list):
    mappings = pd.read_csv(DATA_DIR / "gene_mappings.csv", delimiter=',')
    ensemble = mappings.loc[mappings["symbol"].isin(symbol_list), "query"]
    return ensemble

def translate_ensemble(ensemble_list):
    mappings = pd.read_csv(DATA_DIR / "gene_mappings.csv", delimiter=',')
    symbols = mappings.loc[mappings["query"].isin(ensemble_list), "symbol"]
    return symbols

def filter_genes_by_symbols(X, symbol_list, exclude=False):
    ids = translate_symbols(symbol_list)
    if exclude:
        return X.loc[:, ~X.columns.isin(ids)]
    return X.loc[:, X.columns.isin(ids)]

def search_ndex(search_string, size=100):
    client = ndex2.client.Ndex2("http://public.ndexbio.org")
    search_results = client.search_networks(
        search_string=search_string, # example: 'Alzheimers nodeCount:[10 TO 55]'
        start=0,             # Number of initial records to skip (useful for pagination)
        size=size           # Maximum number of network results to return
    )
    # Returns a dictionary containing a list of network summaries
    # Change that into a dataframe
    networks = pd.DataFrame(search_results.get('networks', []))

def get_ndex_network_by_id(network_uuid):
    client = ndex2.client.Ndex2("http://ndexbio.org")
    client_resp = client.get_network_as_cx2_stream(network_uuid)
    # Create CX2Network factory
    factory = RawCX2NetworkFactory()
    # Convert downloaded network to CX2Network object
    cx2_network = factory.get_cx2network(json.loads(client_resp.content))
    return cx2_network

def get_node_names(cx2_network, name_field="name"):
    node_names = []
    for id, node in cx2_network.get_nodes().items():
        properties = node['v']
        name = properties[name_field]
        node_names.append(name)
    return node_names
