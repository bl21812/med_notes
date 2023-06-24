'''
Notes

Medical meadow mediQA dataset has 2208 entries
columns are 'input', 'instruction', and 'output'
and is by default one train split only

long sequences are currently split - but to train with those we need
    SOAP for each split section !! (i split first then u label)
'''

import os
import torch
import pandas as pd

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from qa_model import QA_Model
from embedder import Embedder
from data_utils import tokenize_qa

tokenizer_source = "medalpaca/medalpaca-13b"
model_source = "medalpaca/medalpaca-13b"
data_source = "medalpaca/medical_meadow_mediqa"

seq_max_length = 2048  # llama max sequence length
seq_doc_stride = 128  # NOTE: may need to be changed

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
# NOTE: test this in jupyter

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")

'''
TODO

edit single-item tokenization so it doesn't overflow
edit qa model to work from embeddings only (just linear units and head)
write demo: embed max length seqs, then concat embeddings to feed into MLP
'''

# TOKENIZE TARGETS SEPARATELY - otherwise number of rows doesn't match up

# NOTE: row names are only for mediQA rn
ds_train_tokenized = ds_train.map(lambda row: {
    'input_tokens': tokenize_qa(tokenizer, row['instruction'], row['input']), 
    'output_tokens': tokenize_qa(tokenizer, row['output'])
}, batched=True, remove_columns=ds_train.column_names)

ds_val_tokenized = ds_val.map(lambda row: {
    'input_tokens': tokenize_qa(tokenizer, row['instruction'], row['input']), 
    'output_tokens': tokenize_qa(tokenizer, row['output'])
}, batched=True, remove_columns=ds_val.column_names)

ds_test_tokenized = None
if ds_test:
    ds_test_tokenized = ds_test.map(lambda row: {
        'input_tokens': tokenize_qa(tokenizer, row['instruction'], row['input']), 
        'output_tokens': tokenize_qa(tokenizer, row['output'])
    }, batched=True, remove_columns=ds_test.column_names)

# TEMP FOR testing
print(ds_train_tokenized[0].keys())
print(len(ds_train_tokenized[0]['input_tokens']))
print(len(ds_train_tokenized[0]['input_tokens'][0]))
print(len(ds_train_tokenized[0]['output_tokens']))
print(len(ds_train_tokenized[0]['output_tokens'][0]))

for tokens in ds_train_tokenized[0]['output_tokens']:
    print(tokenizer.decode(tokens))

quit()

'''
DEMO
Run this to show some inference results
'''

# TODO: Load in model
if os.path.exists(model_source):
    model = None
else:
    model = QA_Model(model_source)

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
