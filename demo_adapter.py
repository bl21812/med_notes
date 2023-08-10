import os
import torch
import numpy as np
import pandas as pd

import evaluate
from datasets import Dataset
from transformers.adapters import ParallelConfig, AdapterTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

tokenizer_source = "knkarthick/meeting-summary-samsum"
base_model_source = "knkarthick/meeting-summary-samsum"
adapter_path = "summ_adapter/2023-08-07/"
adapter_type = "parallel"

data_source = "PARTIAL_half_page_summ_dummy.csv"

input_key = 'transcript'
output_key = 'output'

def tokenize_summary_subsection(tokenizer, dialogue, summary):
    '''
    Returns dict, with important keys:
        input_ids: tokenized dialogue (token IDs)
        labels: tokenized summary (token IDs)
    '''

    dialogue = "summarize: \n\n" + dialogue
    res = tokenizer(dialogue, return_tensors='pt', padding='max_length', max_length=512, truncation=True)

    if summary:
        labels = tokenizer(summary, return_tensors='pt', padding='max_length', max_length=256, truncation=True)['input_ids']
    else:
        labels = torch.tensor([[tokenizer.eos_token_id]])
    res['labels'] = labels

    res = {key: torch.squeeze(item, 0) for key, item in res.items()}

    return res

# ----- MODEL LOADING -----

base_model_source = "knkarthick/meeting-summary-samsum"
m = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_source,
    device_map='auto'
)

adapter_name = m.load_adapter(adapter_path, config=adapter_type)
m.set_active_adapters(adapter_name)

m.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")

# ----- PREPARE DATASET -----

if os.path.exists(data_source):
    if '.csv' in data_source:
        df = pd.read_csv(data_source)
    ds = Dataset.from_pandas(df)

ds_tokenized = ds.map(
    lambda row: tokenize_summary_subsection(
        tokenizer=tokenizer,
        dialogue=row[input_key],
        summary=row[output_key]
    ),
    remove_columns=ds.column_names
)

# ----- INFERENCE -----

idx = 0
inp = ''

while idx < len(ds_tokenized):

    data = ds_tokenized[idx]['input_ids']
    label = ds_tokenized[idx]['labels']

    decoded = tokenizer.decode(data)
    decoded_label = tokenizer.decode(label)

    with torch.no_grad():
        outputs = m.generate(torch.tensor([data]).to('cuda'), max_new_tokens=64)

    print('\n\nDIALOGUE: \n')
    print(decoded)
    print('\n\nEXPECTED OUTPUT: \n')
    print(decoded_label)
    print('\n\nPREDICTED OUTPUT: \n')
    print(tokenizer.batch_decode(outputs)[0])

    idx += 1

    inp = input()
    if not (inp == ''):
        break

print('Bye :D')