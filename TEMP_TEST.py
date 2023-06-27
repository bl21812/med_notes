import pandas as pd
from datasets import Dataset

df = pd.read_csv('soap_ds.csv')
ds = Dataset.from_pandas(df)

print(ds[0])
print(ds[1])
print(ds[2])