import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

import numpy as np

tokenizer_source = "distilbert-base-uncased"
feature_extractor_source = "distilbert-base-uncased"
num_hidden_layers = 3
hidden_dim = 256
feature_extractor_output_dim = 6 * 768

data_source = "placeholder_soap_ds.csv"  # UPDATE WITH CLASS DATASET

input_key = 'note'
label_key = 'class'
num_classes = 4

label_mapping = {
    'S': 0, 
    'O': 1,
    'A': 2,
    'P': 3
}

seed = 0
val_prop = 0.1

save_path = 'soap_class/C0001'

batch_size = 64

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
    features = feature_extractor(**tokens)
    return features['last_hidden_state']

def apply_preprocessing_batch(rows):
    rows[input_key] = [embed_from_text(text) for text in rows[input_key]]
    rows[label_key] = [label_mapping[c] for c in rows[label_key]]
    return rows

# tokenize and embed and 
ds_embeddings = ds.shuffle(seed=seed).map(
    apply_preprocessing_batch, batched=True, batch_size=8
)

# train test split
ds_embeddings = ds_embeddings.train_test_split(test_size=val_prop)
ds_train = ds_embeddings['train']
ds_val = ds_embeddings['test']

# ----- CLASSIFICATION HEAD -----
class_head = torch.nn.Sequential([torch.nn.Flatten()])

# hidden layers
for i in range(num_hidden_layers):
    in_features = feature_extractor_output_dim if (i == 0) else hidden_dim
    class_head.append(torch.nn.Linear(in_features=in_features, out_features=hidden_dim, bias=True))
    class_head.append(torch.nn.ReLU())

# head
in_features = feature_extractor_output_dim if (num_hidden_layers == 0) else hidden_dim
class_head.append(torch.nn.Linear(in_features=in_features, out_features=num_classes, bias=True))
class_head.append(torch.nn.Softmax())

# ----- TRAIN LOOP -----

'''text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = feature_extractor(**encoded_input)'''
