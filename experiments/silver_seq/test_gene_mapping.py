import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


mappings = pd.read_csv(DATA_DIR / "gene_mappings.csv", delimiter=',')

print(mappings.head(10))

test_ids_string = '''PRKACB
CHKA
PRKACA
PRKACG
SUFU
SMO
GLI1
BOC
CDON
PRKAR2A
PRKAR1A
PRKAR2B
PRKAR1B'''

test_ids = test_ids_string.split('\n')

filtered = mappings.loc[mappings["symbol"].isin(test_ids), "query"]

print(filtered)