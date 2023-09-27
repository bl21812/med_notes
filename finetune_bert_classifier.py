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
num_hidden_layers = 1
hidden_dim = 32
feature_extractor_output_dim = pad_seq_length * 768

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

seed = 0
val_prop = 0.1

model_name = 'C0005'
save_path = f'soap_class/{model_name}/'
save_best_only = True

batch_size = 4
epochs = 30
lr = 0.001
weight_decay_lambda = 0.1

# ----- MODEL LOADING -----
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_source)
feature_extractor = DistilBertModel.from_pretrained(feature_extractor_source)

# ----- PREPROCESSING -----
# data loading, tokenization, and embedding

# load
if os.path.exists(data_source):
    if '.csv' in data_source:
        df = pd.read_csv(data_source)
    df.drop_duplicates(subset=[input_key], inplace=True)
    ds = Dataset.from_pandas(df)

# convert to label col (for stratification)
ds = ds.class_encode_column(label_key)

# train test split
ds = ds.shuffle(seed=seed).train_test_split(test_size=val_prop, stratify_by_column=label_key)
ds_train = ds['train']
ds_val = ds['test']

# convert back to string labels
def label_to_str(row):
    return {
        row[input_key], 
        ds_train.features[label_key].int2str(row[label_key])
    }

ds_train = ds_train.map(label_to_str)
ds_val = ds_val.map(label_to_str)

# preprocess (as per how feature extractor was trained)
# lowercase and remove punctuation - i think thats itk
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

# tokenize and embed
ds_train = ds_train.map(
    apply_preprocessing_batch, batched=True, batch_size=8
)
ds_val = ds_val.map(
    apply_preprocessing_batch, batched=True, batch_size=8
)

print(ds_train.features)
print(ds_train[0])
quit()

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

# get class counts for weighting (as per sklearn class weighting)
class_counts = [0 for _ in range(num_classes)]
for row in ds_train:
    class_counts[np.argmax(row[label_key])] += 1
class_weights = [len(ds) / (num_classes * count) for count in class_counts]

# loss and optimizer
loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
optimizer = torch.optim.SGD(class_head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_lambda)

# ----- TRAIN LOOP -----

epoch_val_correct = []
epoch_val_total = []
epoch_val_f1 = []
epoch_losses = []
best_model = None

for epoch in range(epochs):

    print(f'Starting epoch {epoch}')

    inputs = []
    labels = []

    epoch_loss = 0

    val_correct = 0
    val_total = 0

    num_train_batches = 0

    # train
    for row in ds_train:

        # add to batch
        inputs.append(row[input_key])
        labels.append(row[label_key])

        # if batch = batch size, take train step
        if len(inputs) == batch_size:

            num_train_batches += 1

            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)

            optimizer.zero_grad()

            outputs = class_head(inputs)
            l = loss(outputs, labels)
            '''print(outputs)
            print('\n\n')
            print(labels)
            print('\n\n')
            print(l)
            inp = input()
            if not (inp == ''):
                quit()'''
            l.backward()
            optimizer.step()

            epoch_loss += l.item()

            # reset batch
            inputs = []
            labels = []

    # track loss
    epoch_losses.append(epoch_loss / num_train_batches)

    # eval
    epoch_labels = []
    epoch_preds = []

    for row in ds_val:

        inputs = torch.tensor([row[input_key]])
        labels = torch.tensor([row[label_key]])

        with torch.no_grad():

            output = class_head(inputs)
            _, pred_class = torch.max(output.data, 1)
            
            label_class = torch.argmax(labels)
            pred_class = torch.flatten(pred_class)

            epoch_labels.append(label_class)
            epoch_preds.append(pred_class)

            val_total += 1
            val_correct += int(pred_class == label_class)  # NOTE: only works for single input rn

            '''print(f'EPOCH {epoch} EVAL:')
            print(labels)
            print(output)
            print(pred_class)
            print(f'{val_correct} / {val_total} correct')'''

    # f1 score
    epoch_labels = torch.tensor(epoch_labels)
    epoch_preds = torch.tensor(epoch_preds)
    epoch_val_f1.append(f1_score(epoch_labels, epoch_preds, average='macro'))

    # save model if applicable
    if save_best_only:
        if len(epoch_val_f1) == 1:
            best_model = copy.deepcopy(class_head)
        elif epoch_val_f1[-1] > max(epoch_val_f1[:-1]):
            best_model = copy.deepcopy(class_head)

    epoch_val_correct.append(val_correct)
    epoch_val_total.append(val_total)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if best_model:
    torch.save(best_model, f'{save_path}{model_name}.pt')
else:
    torch.save(class_head, f'{save_path}{model_name}.pt')

# plots
xs = [i for i in range(1, epochs+1)]

plt.plot(xs, epoch_losses)
plt.savefig(f'{save_path}train_loss.png')

plt.clf()
plt.plot(xs, np.divide(epoch_val_correct, epoch_val_total))
plt.savefig(f'{save_path}val_acc.png')

plt.clf()
plt.plot(xs, epoch_val_f1)
plt.savefig(f'{save_path}val_f1.png')

'''text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = feature_extractor(**encoded_input)'''
