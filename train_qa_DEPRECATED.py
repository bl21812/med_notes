'''
Notes

Medical meadow mediQA dataset has 2208 entries
columns are 'input', 'instruction', and 'output'
and is by default one train split only

long sequences are currently split - but to train with those we need
    SOAP for each split section !! (i split first then u label)

CURRENTLY USING THE WHOLE DECODER
    just tacking on a custom MLP head
'''

import numpy as np

import os
import torch
import pandas as pd

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from qa_model import QA_Head
from embedder import Embedder
from data_utils import tokenize_qa

llm_version = "medalpaca/medalpaca-13b"  # always taken for embeddings
tokenizer_source = "medalpaca/medalpaca-13b"
model_source = "medalpaca/medalpaca-13b"
data_source = "medalpaca/medical_meadow_mediqa"

seq_max_length = 2048  # llama max sequence length
seq_doc_stride = 128  # NOTE: may need to be changed

num_attention_units = 40
fc_layers = 1
latent_dims = 5120  # each embedding is about 1.2 million features

val_prop = 0.1
test_prop = 0

batch_size = 1
lr = 2e-5  # NOTE: may need to be increased
epochs = 3  # NOTE: may need to be increased
decay = 0.01  # NOTE: idk what val

model_save_name = 'mediQA_finetuned'

# Load data 
# TODO: Add splits for custom loading
if os.path.exists(data_source):
    df = pd.read_csv(data_source)
    ds = Dataset.from_pandas(df)
else:
    ds_train = load_dataset(data_source, split='train[:{}%]'.format(int(100 * (1 - val_prop - test_prop))))
    ds_val = load_dataset(data_source, split='train[{}%:{}%]'.format(int(100 * (1 - val_prop - test_prop)), int(100 * (1 - test_prop))))
    ds_test = None
    if test_prop:
        ds_test = load_dataset(data_source, split='train[{}%:]'.format(int(100 * (1 - test_prop))))

# Preprocessing (including tokenization)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")

'''
TODO

edit qa model to work from embeddings only (just linear units and head)
write demo: embed max length seqs, then concat embeddings to feed into MLP
'''

# OUTPUT SHOULD BE TRUNCATED WITHOUT ANY DOC STRIDE - since it's considered one thing

# NOTE: row names are only for mediQA rn
ds_train_tokenized = ds_train.map(lambda row: {
    'input_tokens': tokenize_qa(tokenizer, row['instruction'], row['input'], max_seq_length=seq_max_length, doc_stride=seq_doc_stride), 
    'output_tokens': tokenize_qa(tokenizer, row['output'])
}, remove_columns=ds_train.column_names)

ds_val_tokenized = ds_val.map(lambda row: {
    'input_tokens': tokenize_qa(tokenizer, row['instruction'], row['input'], max_seq_length=seq_max_length, doc_stride=seq_doc_stride), 
    'output_tokens': tokenize_qa(tokenizer, row['output'])
}, remove_columns=ds_val.column_names)

ds_test_tokenized = None
if ds_test:
    ds_test_tokenized = ds_test.map(lambda row: {
        'input_tokens': tokenize_qa(tokenizer, row['instruction'], row['input'], max_seq_length=seq_max_length, doc_stride=seq_doc_stride), 
        'output_tokens': tokenize_qa(tokenizer, row['output'])
    }, remove_columns=ds_test.column_names)

'''
INFO: 
Each DS is: 
[
    {
        'input_tokens': [
            [
                [token_seq_1],
                ...
            ]
        ], 
        'output_tokens': [
            [
                [token_seq_1], 
                ...
            ]
        ],
    }, 
    ... (next item)
]
'''

# Load in model
# TODO: Add code for locally saved model
if os.path.exists(model_source):
    head = None
else:
    head = QA_Head(fc_layers=fc_layers, input_dims=latent_dims)

# if we load this in first it takes up all the memory
embedder = Embedder(llm_version, num_units=num_attention_units)

loss_func = None
optimizer = None

# NOTE: PAUSE
# why don't i just try using the full model (like as is from huggingface)
# and setting only MLP weights to trainable ??
# it also looks like the embedding layer can take in 32001 tokens
    # so maybe i can just ignore truncation
# and also try just using the huggingface trainer as intended
# NOTE: looks like the generate method is actually what gives predicted response tokens
    # and is different from forward ?
    # if default trainer just uses forward the output might have to be smth else
        # or maybe I can set a custom loss ?

# train loop
for epoch in range(epochs):

    for i in range(len(ds_train_tokenized)):

        item = ds_train_tokenized[i]
        inputs = item['input_tokens'][0]  # NOTE: ignore batch dim for now
        outputs = item['output_tokens'][0]
        print(inputs)

        # Forward pass
        # HOW TO DEAL WITH MLP OUTPUT HAVING 237 DIMENSIONS (of sequences) ??
        latents = None
        for input in inputs:
            l = embedder(torch.tensor([input]))[0]  # add batch dim for model input
            l = torch.squeeze(l)
            if not latents:
                latents = l
            else:
                latents = torch.cat((latents, l), axis=-1)
        # FIX THIS PADDING - since latents has 2 dimensions now instead of 1
        # THEN address comment above (under 'Forward pass')
        padding = torch.tensor([0] * (latent_dims - len(latents)))
        latents = torch.cat(latents, padding)  # pad to length
        latents = latents[None, :]  # add batch dim
        print(latents)
        quit()
        preds = head(latents)
        print(preds)

        quit()

        # Compute loss
        # concat outputs with each other if applicable
        # and compute loss
        loss = None

        # Update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for i in range(len(ds_val_tokenized)):
        print()

# NO TO THE BELOW
args = TrainingArguments(
    model_save_name,
    evaluation_strategy = "epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=decay,
    push_to_hub=True,  # NOTE: remove if causing issues
)

trainer = Trainer(
    model, 
    args, 
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=default_data_collator,  # NOTE: may not work
    tokenizer=tokenizer,  # NOTE: not sure if i need this since it's tokenized above ?
)

trainer.save_model(model_save_name)
