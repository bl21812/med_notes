'''
Script to run inference using a given model and dataset
and view outputs step by step

PEFT DOES NOT WORK WITH CPU OFFLOAD
I need 3 of the GPUs to load 7b or 13b
'''

import os
import torch
import random
import pandas as pd
import numpy as np

from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator, AutoModelForCausalLM, AutoConfig, \
    AutoModelForQuestionAnswering, pipeline, LlamaForCausalLM, GenerationConfig, LlamaTokenizer
from tokenizers.processors import TemplateProcessing
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

from data_utils import tokenize_qa, preprocess_text
from medalpaca_prompt_handler import DataHandler

seed = 0

# file from prompts/ folder
prompt_template = "prompts/prompt_template_SOAP_2.json"

# one of ["decapoda-research/llama-7b-hf", "medalpaca/medalpaca-13b"]
tokenizer_source = "decapoda-research/llama-7b-hf"

# one of ["tloen/alpaca-lora-7b", "medalpaca/medalpaca-lora-13b-8bit", or local folder with adapter files]
model_source = "sectioned_dummy_finetuned/2023-07-18"

# one of ["decapoda-research/llama-7b-hf", "yahma/llama-7b-hf"]
base_model_source = "decapoda-research/llama-7b-hf"

# one of ["medalpaca/medical_meadow_mediqa", "dialogsum/dialogsum.test.jsonl", "soap_ds.csv"]
data_source = "dummy_separated_small_TEST.csv"  

add_sep_token = False
seq_max_length = 2048  # llama max sequence length
seq_doc_stride = 128  # NOTE: may need to be changed

# Load data 
# TODO: Add splits for custom loading
if os.path.exists(data_source):
    if '.json' in data_source:
        df = pd.read_json(data_source, lines=True)
    elif '.csv' in data_source:
        df = pd.read_csv(data_source)
    else:
        raise ValueError('Please provide either a csv, json, or huggingface dataset!')
    # ONLY IF APPLICABLE
    # df.rename(columns={'transcript': 'dialogue'}, inplace=True)
    ds = Dataset.from_pandas(df)
else:
    ds = load_dataset(data_source, split='train')

print('Dataset loaded!')
    
# Preprocessing (including tokenization)

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_source, device_map="auto")
print('Tokenizer loaded!')

tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

if add_sep_token:
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[SEP]']
    })

data_handler = DataHandler(tokenizer, prompt_template=prompt_template, model_max_length=seq_max_length, train_on_inputs=False)

'''columns = ['dialogue']
task = 'summary'
if 'dialogsum' in data_source:
    columns = ['dialogue']
    task = 'dialogsum'
assert columns and task'''

columns = ['instruction', 'transcript']

ds_tokenized = ds.shuffle(seed=seed).map(
    lambda row: tokenize_qa(
        tokenizer, 
        data_handler.generate_prompt_soap_section(**(preprocess_text(row, columns, add_sep=add_sep_token))),
        max_seq_length=seq_max_length, 
        doc_stride=seq_doc_stride
    ), 
    remove_columns=ds.column_names
)
'''
# NOTE: row names are only for mediQA rn
# NOTE: flipped input and instruction
# NOTE: figure out how to add SEP tokens to separate dialogue
ds_tokenized = ds.shuffle(seed=seed).map(lambda row:
    tokenize_qa(
        tokenizer, 
        # data_handler.generate_prompt(instruction=row['instruction'], input=row['input']),  # stock QA task
        data_handler.generate_prompt_interview(**(preprocess_text(row, columns=['transcript'], add_sep=add_sep_token))),  # interview transcript SOAP task
        max_seq_length=seq_max_length, 
        doc_stride=seq_doc_stride
    ), 
)  # Custom tokenization
'''
print('Tokenization complete!')

'''
OUTDATED !!!
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

# THE COMMENTED SECTION BELOW IS IF NOT USING PEFT
'''config = AutoConfig.from_pretrained(base_model_source)
with init_empty_weights():
    model = LlamaForCausalLM._from_config(config)
model.tie_weights()
max_memory = {0: "0GIB", 1: "10GIB", 2: "10GIB", 3: "10GIB"}  # only last GPU
device_map = infer_auto_device_map(model, max_memory=max_memory)'''

# for my 3 gpu setup
# device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 2, 'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2, 'model.layers.28': 2, 'model.layers.29': 2, 'model.layers.30': 2, 'model.layers.31': 2, 'model.norm': 2, 'lm_head': 0}

model = LlamaForCausalLM.from_pretrained(
    base_model_source,  # change to model_source if not using peft
    load_in_8bit=True,
    device_map='auto', 
    # offload_folder='offload', 
    # llm_int8_enable_fp32_cpu_offload=True,
    torch_dtype=torch.float16
) 

# MOVE THIS ONE TO TRAIN
'''lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=('q_proj', 'v_proj'),
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)'''
model = PeftModel.from_pretrained(
    model=model, 
    model_id=model_source,
    is_trainable=False, 
    torch_dtype=torch.float16
)
# model.half()  # if not peft
model.eval()
'''model = load_checkpoint_and_dispatch(
    model,
    "medalpaca-13b",
    device_map=device_map
)'''

device_map = model.hf_device_map
print(device_map)

# TODO: Expand embeddings to accomodate for SEP
if add_sep_token:
    print('Have not added support for this yet!')

print('Model loaded!')

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

# TEST MODEL WITH CUSTOM INPUT
'''
inp = ''
while True:

    print('Instruction: ')
    instruction = input()
    print()
    print('Input: ')
    inputs = input()
    print()

    prompt = data_handler.generate_prompt(instruction=instruction, input=inputs)

    tokenized = tokenize_qa(tokenizer, prompt, max_seq_length=seq_max_length, doc_stride=seq_doc_stride)

    tokenized_inputs = tokenized['input_ids']

    generation_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4
    )

    with torch.no_grad():
        generate_output = model.generate(
            inputs=torch.tensor([tokenized_inputs]).to('cuda'), 
            generation_config=generation_config,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True
        )
        # print(generate_output)
        # print(generate_ids.size())
        pred = tokenizer.decode(generate_output.sequences[0])
        input_prompt = tokenizer.decode(tokenized_inputs)
        print('---------- INPUT ----------')
        print(input_prompt)
        print()
        print('---------- PREDICTED OUTPUT ----------')
        print(pred)
        print()

    if inp == 'exit':
        quit()
'''

inp = ''
idx = random.randint(0, len(ds_tokenized) - 1)
seen_idx = [idx]

while True:

    item = ds_tokenized[idx]
    inputs = item['input_ids']

    print('Number of tokens in input: {}'.format(len(inputs)))
    print()

    generation_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.75,
        top_k=40,  # higher = more memory
        num_beams=2,  # higher = more memory
        # early_stopping=True, 
        # no_repeat_ngram_size=3  # need to take into account summary contexts! (what is the longest sequence that could repeat)
    )

    # Get model prediction
    with torch.no_grad():
        generate_output = model.generate(
            inputs=torch.tensor([inputs]).to('cuda'), 
            generation_config=generation_config,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True
        )
        # print(generate_output)
        pred = tokenizer.decode(generate_output.sequences[0])
        input_prompt = tokenizer.decode(inputs)
        print('---------- INPUT ----------')
        print(input_prompt)
        print()
        print('---------- PREDICTED OUTPUT ----------')
        print(pred)
        print()

    inp = input()
    if not (inp == ''):
        break

    while idx in seen_idx:
        idx = random.randint(0, len(ds_tokenized) - 1)
    seen_idx.append(idx)


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
