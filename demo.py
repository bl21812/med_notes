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
    AutoModelForQuestionAnswering, pipeline, LlamaForCausalLM, GenerationConfig
from tokenizers.processors import TemplateProcessing
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

from data_utils import tokenize_qa, preprocess_text
from medalpaca_prompt_handler import DataHandler

seed = 0

prompt_template = "prompts/prompt_template_dialogue_summary_2.json"
# tokenizer_source = "medalpaca/medalpaca-13b"
tokenizer_source = "yahma/llama-13b-hf"
model_source = "yahma/alpaca-13b-lora"
# model_source = "medalpaca/medalpaca-lora-13b-8bit"  # pre-trained from hub
# model_source = "dialogsum_finetuned/2023-07-02"  # local checkpoint
base_model_source = "yahma/llama-13b-hf"
# data_source = "medalpaca/medical_meadow_mediqa"  # from hub
data_source = "dialogsum/dialogsum.test.jsonl"

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
    ds = Dataset.from_pandas(df)
else:
    ds = load_dataset(data_source, split='train')

print('Dataset loaded!')
    
# Preprocessing (including tokenization)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")
print('Tokenizer loaded!')

tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

if add_sep_token:
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[SEP]']
    })

data_handler = DataHandler(tokenizer, prompt_template=prompt_template, model_max_length=seq_max_length, train_on_inputs=False)

columns = None
task = None
if 'dialogsum' in data_source:
    columns = ['dialogue']
    task = 'dialogsum'
assert columns and task

ds_tokenized = ds.shuffle(seed=seed).map(
    lambda row: tokenize_qa(
        tokenizer, 
        data_handler.generate_prompt_summary(**(preprocess_text(row, columns, task, add_sep=add_sep_token))),
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
        data_handler.generate_prompt(instruction=row['instruction'], input=row['input']),  # stock QA task
        # data_handler.generate_prompt_interview_s_only(transcript=preprocess_text(row['transcript'], add_sep=add_sep_token)),  # interview transcript SOAP task
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
    is_trainable=False
)
# model.half()  # if not peft
model.eval()
'''model = load_checkpoint_and_dispatch(
    model,
    "medalpaca-13b",
    device_map=device_map
)'''

# TODO: Expand embeddings to accomodate for SEP
if add_sep_token:
    print('Have not added support for this yet!')

print('Model loaded!')

'''
# Test a few examples
tests = [
    'what are the side effects of radiation therapy?',
    'what are the symptoms of a common cold?'
]

for item in tests:
    x = tokenizer(data_handler.generate_prompt(item))
'''

inp = ''
idx = random.randint(0, len(ds_tokenized) - 1)
seen_idx = [idx]

while True:

    item = ds_tokenized[idx]
    # inputs = item['input_tokens']
    inputs = item['input_ids']
    # true_output = item['output']
    # transcript = item['transcript']
    # instruction = item['instruction']
    # context = item['input']

    '''if len(inputs) > 3500:  # don't have enough memory for huge samples lol
        if idx not in seen_idx:
            seen_idx.append(idx)
        while idx in seen_idx:
            idx = random.randint(0, len(ds_tokenized) - 1)
        continue'''

    '''print('---------- INSTRUCTION ----------')
    print(instruction)
    print()
    print('---------- CONTEXT ----------')
    print(context)
    # print('---------- TRANSCRIPT ----------')
    # print(repr(transcript))
    # print()
    print('---------- EXPECTED OUTPUT ----------')
    print(true_output)
    print()'''
    print('Number of tokens in input: {}'.format(len(inputs)))
    print()

    # Get model prediction
    generation_config = GenerationConfig(max_new_tokens=1024)
    with torch.no_grad():
        generate_ids = model.generate(
            inputs=torch.tensor([inputs]).to('cuda'), 
            generation_config=generation_config,
            max_new_tokens=256
        )
        print(generate_ids)
        print(generate_ids.size())
        pred = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
