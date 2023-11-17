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
adapter_path = "summ_adapter/0013/"
adapter_type = "parallel"

data_source = "abbrev_75_20_both_nov_16.csv"
scrub_transcripts = True

input_key = 'transcript'
output_key = 'output'

# just indicates how much of the ds was used for train
# this demo only uses the latter (val_prop) proportion of the ds
val_prop = 0.1

# seed used for ds shuffle in train
seed = 0

def scrub_all(text):
    '''
    remove newlines, speaker indicators, punctuation (periods, commas, question marks)
    '''
    text = repr(text).replace('\\n', '')
    text = text.replace('D:', '')
    text = text.replace('P:', '')
    text = text.replace('Doctor:', '')
    text = text.replace('Patient:', '')
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.replace('?', '')
    text = text.replace('!', '')
    text = text.replace(':', '')
    return text

def tokenize_summary_subsection(tokenizer, dialogue, summary):
    '''
    Returns dict, with important keys:
        input_ids: tokenized dialogue (token IDs)
        labels: tokenized summary (token IDs)
    '''

    dialogue = scrub_all(dialogue) if scrub_transcripts else dialogue
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
    # Remove empty rows
    df.drop(df.index[df[input_key].isna()], inplace=True)
    ds = Dataset.from_pandas(df)

ds_tokenized = ds.shuffle(seed=seed).map(
    lambda row: tokenize_summary_subsection(
        tokenizer=tokenizer,
        dialogue=row[input_key],
        summary=row[output_key]
    ),
    remove_columns=ds.column_names
)

# ----- INFERENCE -----

idx = int(len(ds_tokenized) * (1 - val_prop))
inp = ''

while idx < len(ds_tokenized):

    data = ds_tokenized[idx]['input_ids']
    label = ds_tokenized[idx]['labels']

    decoded = tokenizer.decode(data, skip_special_tokens=True)
    decoded_label = tokenizer.decode(label, skip_special_tokens=True)

    with torch.no_grad():
        outputs = m.generate(torch.tensor([data]).to('cuda'), max_new_tokens=100)

    print('\n\nDIALOGUE: \n')
    print(decoded)
    print('\n\nEXPECTED OUTPUT: \n')
    print(decoded_label)
    print('\n\nPREDICTED OUTPUT: \n')
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

    idx += 1

    inp = input()
    if not (inp == ''):
        break

print('Bye :D')
