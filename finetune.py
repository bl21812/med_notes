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
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator, LlamaForCausalLM, \
    DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, get_peft_model_state_dict

from qa_model import QA_Head
from embedder import Embedder

from data_utils import tokenize_qa, preprocess_text
from medalpaca_prompt_handler import DataHandler

seed = 0

tokenizer_source = "medalpaca/medalpaca-13b"
model_source = "medalpaca/medalpaca-lora-13b-8bit"
base_model_source = "decapoda-research/llama-13b-hf"

prompt_template = "prompts/prompt_template_dialogue_summary.json"
data_source = "dialogsum.train.jsonl"
data_source_train = "dialogsum.train.jsonl"
data_source_eval = "dialogsum.test.jsonl"

add_sep_token = False
seq_max_length = 2048  # llama max sequence length
seq_doc_stride = 128  # NOTE: may need to be changed

num_attention_units = 40
fc_layers = 1
latent_dims = 5120  # each embedding is about 1.2 million features

val_prop = 0.1
test_prop = 0

num_devices = 3

# Hparams
batch_size = 3
optim = 'adafactor'
lr = 2e-5
lr_scheduler_type = 'cosine'
epochs = 3
decay = 0.01
warmup_steps = 200
eval_steps = 200  # num of steps between evals

model_save_name = 'dialogsum_finetuned/2023-07-02'

# Load data 
# TODO: Add test support
if os.path.exists(data_source):
    if '.json' in data_source:
        df_train = pd.read_json(data_source_train, lines=True)
        df_val = pd.read_json(data_source_eval, lines=True)
    elif '.csv' in data_source:
        df_train = pd.read_csv(data_source_train)
        df_val = pd.read_csv(data_source_eval)
    else:
        raise ValueError('Please provide either a csv, json, or huggingface dataset!')
    ds_train = Dataset.from_pandas(df_train)
    ds_val = Dataset.from_pandas(df_val)
    ds_test = None
else:
    ds_train = load_dataset(data_source, split='train[:{}%]'.format(int(100 * (1 - val_prop - test_prop))))
    ds_val = load_dataset(data_source, split='train[{}%:{}%]'.format(int(100 * (1 - val_prop - test_prop)), int(100 * (1 - test_prop))))
    ds_test = None
    if test_prop:
        ds_test = load_dataset(data_source, split='train[{}%:]'.format(int(100 * (1 - test_prop))))

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

# OUTPUT SHOULD BE TRUNCATED WITHOUT ANY DOC STRIDE - since it's considered one thing

# TODO: make called prompt function dynamic w.r.t. task

train_columns = None
val_columns = None
test_columns = None
task = None
if 'dialogsum' in data_source:
    train_columns = ['dialogue', 'summary']
    val_columns = ['dialogue', 'summary1']
    task = 'dialogsum'
assert (train_columns and task)

# TODO: do i need an attention mask ??

'''ds_train_tokenized = ds_train.shuffle(seed=seed).map(lambda row: {
    'input_ids': tokenize_qa(
        tokenizer, 
        data_handler.generate_prompt_summary(**(preprocess_text(row, train_columns, task, add_sep=add_sep_token))),
        max_seq_length=seq_max_length, 
        doc_stride=seq_doc_stride
    )
}, remove_columns=ds_train.column_names)'''

ds_train_tokenized = ds_train.shuffle(seed=seed).map(
    lambda row: tokenize_qa(
        tokenizer, 
        data_handler.generate_prompt_summary(**(preprocess_text(row, train_columns, task, add_sep=add_sep_token))),
        max_seq_length=seq_max_length, 
        doc_stride=seq_doc_stride
    ), 
    remove_columns=ds_train.column_names
)

'''ds_val_tokenized = ds_val.shuffle(seed=seed).map(lambda row: {
    'input_ids': tokenize_qa(
        tokenizer, 
        data_handler.generate_prompt_summary(**(preprocess_text(row, val_columns, task, add_sep=add_sep_token))),
        max_seq_length=seq_max_length, 
        doc_stride=seq_doc_stride
    )
}, remove_columns=ds_val.column_names)'''

ds_val_tokenized = ds_val.shuffle(seed=seed).map(
    lambda row: tokenize_qa(
        tokenizer, 
        data_handler.generate_prompt_summary(**(preprocess_text(row, val_columns, task, add_sep=add_sep_token))),
        max_seq_length=seq_max_length, 
        doc_stride=seq_doc_stride
    ), 
    remove_columns=ds_val.column_names
)

ds_test_tokenized = None
if ds_test:
    '''ds_test_tokenized = ds_test.shuffle(seed=seed).map(lambda row: {
        'input_ids': tokenize_qa(
            tokenizer, 
            data_handler.generate_prompt_summary(**(preprocess_text(row, test_columns, task, add_sep=add_sep_token))),
            max_seq_length=seq_max_length, 
            doc_stride=seq_doc_stride
        )
    }, remove_columns=ds_test.column_names)'''
    ds_test_tokenized = ds_test.shuffle(seed=seed).map(
        lambda row: tokenize_qa(
            tokenizer, 
            data_handler.generate_prompt_summary(**(preprocess_text(row, test_columns, task, add_sep=add_sep_token))),
            max_seq_length=seq_max_length, 
            doc_stride=seq_doc_stride
        ), 
        remove_columns=ds_test.column_names
    )

print('Preprocessing complete!')

'''
THIS IS OUTDATED !! (i'm just following medalpaca train script now)
INFO: 
Each DS is: 
[
    {
        'input_ids': [
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
    model = LlamaForCausalLM.from_pretrained(
        base_model_source,  # change to model_source if not using peft
        load_in_8bit=True,
        device_map='auto', 
        # offload_folder='offload', 
        # llm_int8_enable_fp32_cpu_offload=True,
        torch_dtype=torch.float16
    ) 
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=('q_proj', 'v_proj'),
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ??
model.is_parallelizable = True
model.model_parallel = True

# TODO: Expand embeddings to accomodate for SEP
if add_sep_token:
    print('Have not added support for this yet!')

# Training

trainer_args = TrainingArguments(
    per_device_train_batch_size=batch_size // 3,
    per_device_eval_batch_size=batch_size // 3,
    gradient_accumulation_steps=num_devices,
    warmup_steps=warmup_steps,
    num_train_epochs=epochs,
    learning_rate=lr,
    fp16=True,
    bf16=False,
    logging_steps=10,
    optim=optim,
    lr_scheduler_type=lr_scheduler_type,
    evaluation_strategy = "steps",
    save_strategy="steps",
    eval_steps=eval_steps,
    save_steps=eval_steps,
    output_dir=model_save_name,
    load_best_model_at_end=True,
    ddp_find_unused_parameters=None,
    group_by_length=False,
    fsdp="",
    fsdp_transformer_layer_cls_to_wrap=None
)

trainer = Trainer(
    model, 
    args=trainer_args, 
    train_dataset=ds_train_tokenized,
    eval_dataset=ds_val_tokenized,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
)

model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

trainer.train()

model.save_pretrained(model_save_name)
