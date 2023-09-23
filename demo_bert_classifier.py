import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

import numpy as np

tokenizer_source = "distilbert-base-uncased"
feature_extractor_source = "distilbert-base-uncased"
pad_seq_length = 32

data_source = "soap_class_ds.csv"

input_key = 'note'
label_key = 'class'
num_classes = 4

label_mapping = {
    'S': [1., 0., 0., 0.], 
    'O': [0., 1., 0., 0.],
    'A': [0., 0., 1., 0.],
    'P': [0., 0., 0., 1.]
}

model_name = 'C0002'
save_path = f'soap_class/{model_name}/'
model_path = f'{save_path}{model_name}.pt'

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
    text = text.replace('- ', '')
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.replace('?', '')
    text = text.replace('!', '')
    text = text.replace(':', '')
    return text.lower()

# get embeddings from text input (basically apply all preprocessing needed)
def embed_from_text(text):
    text = preprocess_str(text)
    tokens = tokenizer(text, return_tensors='pt', padding='max_length', max_length=pad_seq_length)
    features = feature_extractor(**tokens)
    return features['last_hidden_state']

def apply_preprocessing_batch(rows):
    rows[input_key] = [embed_from_text(text) for text in rows[input_key]]
    rows[label_key] = [label_mapping[c] for c in rows[label_key]]
    return rows

ds_orig = copy.deepcopy(ds)

# tokenize and embed
ds_embeddings = ds.map(
    apply_preprocessing_batch, batched=True, batch_size=8
)

# ----- LOAD TRAINED HEAD -----
class_head = torch.load(model_path)
class_head.eval()

# ----- INFERENCE -----
inp = ''

for row, orig_row in zip(ds_embeddings, ds_orig):

    print(orig_row[input_key])

    input = torch.tensor([row[input_key]])
    output = class_head(input)
    print(output)
    pred_class = torch.argmax(output)
    print(pred_class)
    label_onehot = row[label_key]
    print(label_onehot)
    label = torch.argmax(label_onehot)
    print(label)

    inp = input()
    if not (input == ''):
        quit()
