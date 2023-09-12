import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

tokenizer_source = "distilbert-base-uncased"
feature_extractor_source = "distilbert-base-uncased"
num_hidden_layers = 3
hidden_dim = 256

data_source = "placeholder_soap_ds.csv"  # UPDATE WITH CLASS DATASET

input_key = 'note'
label_key = 'class'

seed = 0
val_prop = 0.1

save_path = 'soap_class/C0001'

# ----- MODEL LOADING -----
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_source)
feature_extractor = DistilBertModel.from_pretrained(feature_extractor_source)

# ----- PREPROCESSING -----
# data loading, tokenization, and embedding

# load
if os.path.exists(data_source):
    if '.csv' in data_source:
        df = pd.read_csv(data_source)
    ds = Dataset.from_pandas(df)

# preprocess (as per how feature extractor was trained)
# lowercase and remove punctuation - i think thats it
def preprocess_str(text):
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.replace('?', '')
    text = text.replace('!', '')
    text = text.replace(':', '')
    return text.lower()

# get embeddings from text input (basically apply all preprocessing needed)
def embed_from_text(text):
    text = preprocess_str(text)
    tokens = tokenizer(text, return_tensors='pt')
    return feature_extractor(**tokens)

def apply_preprocessing_row(row):
    row[input_key] = embed_from_text(row[input_key])
    return row

# tokenize and embed
ds_embeddings = ds.shuffle(seed=seed).map(apply_preprocessing_row)

# train test split
ds_embeddings = ds_embeddings.train_test_split(test_size=val_prop)
ds_train = ds_embeddings['train']
ds_val = ds_embeddings['test']

print(ds_train[0])
print('\n\n')
quit()

'''text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = feature_extractor(**encoded_input)'''
