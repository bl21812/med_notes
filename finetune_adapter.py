import os
import torch
import pandas as pd

from datasets import Dataset
from transformers.adapters import ParallelConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from data_utils import tokenize_summary_subsection

tokenizer_source = "knkarthick/meeting-summary-samsum"
base_model_source = "knkarthick/meeting-summary-samsum"

data_source = "test_ds_summ.csv"

input_key = 'transcript'
output_key = 'output'

seed = 0
val_prop = 0.2

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# ----- MODEL LOADING -----

# believe i can use this instead of AutoAdapterModel ?
model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_source, 
    device_map='auto'
)

# idk if parallel adapter is good for few shot
config = ParallelConfig(
    mh_adapter=True,
    output_adapter=True,  # can keep both of these in for now (unsure if needed)
    reduction_factor=16,  # important param !! (not sure what val)
    non_linearity="relu"
)
model.add_adapter("bottleneck_adapter", config=config)

model.train_adapter("bottleneck_adapter")
model.set_active_adapters("bottleneck_adapter")
print_trainable_parameters(model)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")
print ('Tokenizer loaded!')

# ----- PREPARE DATASET -----

if os.path.exists(data_source):
    if '.csv' in data_source:
        df = pd.read_csv(data_source)
    ds = Dataset.from_pandas(df)

ds_tokenized = ds.shuffle(seed=seed).map(
    lambda row: tokenize_summary_subsection(
        tokenizer=tokenizer,
        dialogue=row[input_key],
        summary=row[output_key]
    ), 
    remove_columns=ds.column_names
)

ds_tokenized = ds_tokenized.train_test_split(test_size=val_prop)
ds_train_tokenized = ds_tokenized['train']
ds_val_tokenized = ds_tokenized['test']

print('Preprocessing complete!')

elem1 = ds_train_tokenized[0]
print(tokenizer.decode(elem1['input_ids']))
print(tokenizer.decode(elem1['labels']))

# ----- TRAINING -----

# KEEPING THE BELOW AS A REFERENCE FOR WORKING GENERATION
'''
example = """
summarize:

D: OK. Your lymph nodes don’t feel swollen to me, which is a good sign. So here’s what I think should be our next steps. I’m going to order an exercise stress test for you to do, which will help figure out if there’s anything wrong with your heart. I’m also going to order bloodwork to rule out any possible infection that might be causing your chest pain. In the meantime, I’m going to prescribe you 2 pills of aspirin to take as needed when you feel that chest pain, and we’ll see if that helps relieve the pain. And let’s follow up once we get all the test results back. My office will contact you to set up an appointment in a few weeks. How does that sound? 

P: Sounds great Doc. Thanks!
"""

tokenized = tokenizer(example, return_tensors='pt')['input_ids']
# print(tokenized)

# decoded = tokenizer.decode(tokenized)
# print(decoded)

with torch.no_grad():
    outputs = model.generate(
        tokenized.to('cuda'),
        max_new_tokens=64
    )  # input shouldnt be a list ??

print(outputs)
print(tokenizer.batch_decode(outputs)[0])
'''
