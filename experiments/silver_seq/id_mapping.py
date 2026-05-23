import mygene
import pandas as pd

mg = mygene.MyGeneInfo()

counts_path = '/Users/idekeradmin/Dropbox/GitHub/AD_prediction_blood/experiments/silver_seq/silver_seq_counts.txt'
silver_seq_counts = pd.read_csv(counts_path,  sep="\t")
silver_seq_counts = silver_seq_counts.astype(int)

# Bulk query — pass a list of IDs
gene_list = silver_seq_counts.index.tolist()
results = mg.querymany(gene_list, scopes='ensembl.gene',
                       fields='symbol,name,entrezgene',
                       species='human', as_dataframe=True)

# Save locally
results.to_csv('gene_mappings.csv')