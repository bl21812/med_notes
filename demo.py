'''
Script to run inference using a given model and dataset
and view outputs step by step
'''

import os
import torch
import random
import pandas as pd
import numpy as np

# from peft import PeftModel
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator, AutoModelForCausalLM, AutoConfig, \
    AutoModelForQuestionAnswering, pipeline, LlamaForCausalLM, GenerationConfig
from tokenizers.processors import TemplateProcessing
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

from data_utils import tokenize_qa, preprocess_text
from medalpaca_prompt_handler import DataHandler

prompt_template = "prompt_template_SOAP.json"
tokenizer_source = "medalpaca/medalpaca-13b"
model_source = "medalpaca/medalpaca-13b"
base_model_source = "decapoda-research/llama-13b-hf"
# data_source = "medalpaca/medical_meadow_mediqa"
data_source = 'soap_ds.csv'

seq_max_length = 32001  # llama max sequence length  # ORIGINAL 2048
seq_doc_stride = 128  # NOTE: may need to be changed

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")
print('Tokenizer loaded!')

# print(tokenizer.get_post_processor())

tokenizer.add_special_tokens({
    'additional_special_tokens': ['[CLS]', '[SEP]']
})

print(tokenizer.get_vocab()["[CLS]"])
print(tokenizer.get_vocab()["[SEP]"])

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.get_vocab()["[CLS]"]),
        ("[SEP]", tokenizer.get_vocab()["[SEP]"]),
    ],
)

inp = 'Hello, how are you doing today? [SEP] Good, how are you? [SEP] Pneumonia halitosis cardiac arrest. [SEP] Agreed, I do have those.'
tokenized = tokenizer(inp, add_special_tokens=True)
print(tokenized)

exit()

# Load data 
# TODO: Add splits for custom loading
if os.path.exists(data_source):
    df = pd.read_csv(data_source)
    ds = Dataset.from_pandas(df)
else:
    ds = load_dataset(data_source, split='train')

print('Dataset loaded!')

# THIS DIDNT WORK FOR SOME REASON LOL
'''
use_default_pipeline = False
if use_default_pipeline:

    qa_pipeline = pipeline('question-answering', model=model_source, tokenizer=model_source)

    inp = ''
    idx = 0

    while True:

        item = ds[idx]
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
        pred = qa_pipeline({'question': instruction, 'context': context})
        print('---------- PREDICTED OUTPUT ----------')
        print(pred)

        inp = input()
        if not (inp == ''):
            break
        idx += 1

    quit()
'''
    
# Preprocessing (including tokenization)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")
print('Tokenizer loaded!')

data_handler = DataHandler(tokenizer, prompt_template=prompt_template, model_max_length=seq_max_length, train_on_inputs=False)

# NOTE: row names are only for mediQA rn
# NOTE: flipped input and instruction
# NOTE: figure out how to add SEP tokens to separate dialogue
ds_tokenized = ds.map(lambda row: {
    'input_tokens': tokenize_qa(
        tokenizer, 
        #data_handler.generate_prompt(instruction=row['input'], input=row['instruction']),  # stock QA task
        data_handler.generate_prompt_interview(transcript=preprocess_text(row['transcript'])),  # interview transcript SOAP task
        max_seq_length=seq_max_length, 
        doc_stride=seq_doc_stride
    ), 
    'transcript': preprocess_text(row['transcript']),
    'output': preprocess_text(row['output'])
})  # Custom tokenization

print('Tokenization complete!')

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
        model = LlamaForCausalLM._from_config(config)
    model.tie_weights()
    max_memory = {0: "0GIB", 1: "0GIB", 2: "0GIB", 3: "8GIB"}  # only last GPU
    device_map = infer_auto_device_map(model, max_memory=max_memory)
    # device_map = {"": 0}  # from medalpaca inferer class
    model = LlamaForCausalLM.from_pretrained(
        model_source,  # change to base_model_source if using peft
        device_map=device_map, 
        offload_folder='offload', 
        torch_dtype=torch.float16
    )  # ideally load in 8bit but doesn't seem to be working on server
    '''model = PeftModel.from_pretrained(
        model,
        model_id=model_source,
        torch_dtype=torch.float16,
        device_map=device_map,
    )'''
    model.half()
    model.eval()
    '''model = load_checkpoint_and_dispatch(
        model,
        "medalpaca-13b",
        device_map=device_map
    )'''

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
    inputs = item['input_tokens']
    true_output = item['output']
    transcript = item['transcript']
    # instruction = item['instruction']
    # context = item['input']

    '''if len(transcript) > 1500:  # don't have enough memory for huge samples lol
        if idx not in seen_idx:
            seen_idx.append(idx)
        while idx in seen_idx:
            idx = random.randint(0, len(ds_tokenized) - 1)
        continue'''

    # print('---------- INSTRUCTION ----------')
    # print(instruction)
    # print()
    # print('---------- CONTEXT ----------')
    # print(context)
    print('---------- TRANSCRIPT ----------')
    print(repr(transcript))
    print()
    print('---------- EXPECTED OUTPUT ----------')
    print(true_output)
    print()
    print('Number of tokens in transcript: {}'.format(len(inputs)))
    print()

    # Get model prediction
    generation_config = GenerationConfig(max_new_tokens=10000)
    with torch.no_grad():
        generate_ids = model.generate(
            torch.tensor([inputs]).to('cuda'), 
            generation_config=generation_config,
            max_new_tokens=10000
        )
        print(generate_ids)
        pred = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
