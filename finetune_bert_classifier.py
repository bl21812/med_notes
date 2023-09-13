import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

import numpy as np

tokenizer_source = "distilbert-base-uncased"
feature_extractor_source = "distilbert-base-uncased"
pad_seq_length = 32
num_hidden_layers = 3
hidden_dim = 256
feature_extractor_output_dim = pad_seq_length * 768

data_source = "placeholder_soap_ds.csv"  # UPDATE WITH CLASS DATASET

input_key = 'note'
label_key = 'class'
num_classes = 4

label_mapping = {
    'S': [1., 0., 0., 0.], 
    'O': [0., 1., 0., 0.],
    'A': [0., 0., 1., 0.],
    'P': [0., 0., 0., 1.]
}

seed = 0
val_prop = 0.1

save_path = 'soap_class/C0001'

batch_size = 4
epochs = 10
lr = 0.001

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
    tokens = tokenizer(text, return_tensors='pt', padding='max_length', max_length=pad_seq_length)
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
class_head = torch.nn.Sequential(torch.nn.Flatten())

# hidden layers
for i in range(num_hidden_layers):
    in_features = feature_extractor_output_dim if (i == 0) else hidden_dim
    class_head.append(torch.nn.Linear(in_features=in_features, out_features=hidden_dim, bias=True))
    class_head.append(torch.nn.ReLU())

# head
in_features = feature_extractor_output_dim if (num_hidden_layers == 0) else hidden_dim
class_head.append(torch.nn.Linear(in_features=in_features, out_features=num_classes, bias=True))
class_head.append(torch.nn.Softmax())

# loss and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(class_head.parameters(), lr=lr, momentum=0.9)

# ----- TRAIN LOOP -----

epoch_val_correct = []
epoch_val_total = []

for epoch in range(epochs):

    inputs = []
    labels = []

    val_correct = 0
    val_total = 0

    # train
    for row in ds_train:

        # add to batch
        inputs.append(row[input_key])
        labels.append(row[label_key])

        # if batch = batch size, take train step
        if len(inputs) == batch_size:

            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)

            optimizer.zero_grad()

            outputs = class_head(inputs)
            print(outputs)
            l = loss(outputs, labels)
            print('\n\n')
            print(labels)
            print('\n\n')
            print(l)
            inp = input()
            if not (inp == ''):
                quit()
            l.backward()
            optimizer.step()

            # TODO: TRACK RUNNING LOSS

            # reset batch
            inputs = []
            labels = []

    # eval
    for row in ds_val:
        input = torch.tensor([row[input_key]])
        label = torch.tensor([row[label_key]])
        with torch.no_grad():
            output = class_head(input)
            _, pred_class = torch.max(output.data, 1)
            val_total += label.size(0)
            val_correct += (pred_class == label).sum().item()

    epoch_val_correct.append(val_correct)
    epoch_val_total.append(val_total)

'''text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = feature_extractor(**encoded_input)'''
