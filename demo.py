'''
Script to run inference using a given model and dataset
and view outputs step by step
'''

import numpy as np

import os
import torch
import pandas as pd

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator, AutoModelForCausalLM, AutoConfig

from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

from data_utils import tokenize_qa

tokenizer_source = "medalpaca/medalpaca-13b"
model_source = "medalpaca/medalpaca-13b"
data_source = "medalpaca/medical_meadow_mediqa"

seq_max_length = 32001  # llama max sequence length  # ORIGINAL 2048
seq_doc_stride = 128  # NOTE: may need to be changed

# Load data 
# TODO: Add splits for custom loading
if os.path.exists(data_source):
    df = pd.read_csv(data_source)
    ds = Dataset.from_pandas(df)
else:
    ds = load_dataset(data_source, split='train')

# Preprocessing (including tokenization)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")

'''
TODO

edit qa model to work from embeddings only (just linear units and head)
write demo: embed max length seqs, then concat embeddings to feed into MLP
'''

# NOTE: row names are only for mediQA rn
ds_tokenized = ds.map(lambda row: {
    'input_tokens': tokenize_qa(tokenizer, row['instruction'], row['input'], max_seq_length=seq_max_length, doc_stride=seq_doc_stride)
})

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
    model = None
else:
    config = AutoConfig.from_pretrained(model_source)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    max_memory = {0: "0GIB", 1: "6GIB", 2: "6GIB", 3: "6GIB"}  # avoid GPU 0
    device_map = infer_auto_device_map(model, max_memory=max_memory)
    model = AutoModelForCausalLM.from_pretrained(model_source, device_map="auto")
    '''model = load_checkpoint_and_dispatch(
        model,
        "",  # where are the checkpoint files ?
        device_map=device_map
    )'''


inp = ''
idx = 0
while True:

    item = ds_tokenized[idx]
    inputs = item['input_tokens'][0]  # NOTE: ignore batch dim for now
    true_output = item['output']
    instruction = item['instruction']
    context = item['input']
    print('---------- INSTRUCTION ----------')
    print(instruction)
    print()
    print('---------- CONTEXT ----------')
    print(context)
    print()
    print('---------- EXPECTED OUTPUT ----------')
    print(true_output)
    print()

    # Get model prediction
    generate_ids = model.generate(torch.tensor([inputs]).to('cuda'), max_new_tokens=500)  # need a max length ?
    pred = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print('---------- PREDICTED OUTPUT ----------')
    print(pred)

    inp = input()
    if not (inp == ''):
        break
    idx += 1


'''
EVERYTHING UNDER IS OLD
'''

quit()

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
