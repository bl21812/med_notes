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
adapter_name = "bottleneck_adapter"

data_source = "PARTIAL_half_page_summ_dummy.csv"

input_key = 'transcript'
output_key = 'output'

seed = 0
val_prop = 0.2

save_path = 'summ_adapter/2023-08-08'

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
    reduction_factor=32,  # important param !! (not sure what val)
    non_linearity="relu"
)
model.add_adapter(adapter_name, config=config)

model.train_adapter(adapter_name)
model.set_active_adapters(adapter_name)
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

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=tokenizer_source)

print('Preprocessing complete!')

# ----- TRAINING -----

rouge = evaluate.load('rouge')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# HPARAMS !!!
# TRAIN LOSS IS NOT LOGGING AT EPOCHS !!
training_args =  Seq2SeqTrainingArguments(
    learning_rate=1e-4,  # apparently this works well
    num_train_epochs=200,  # dunno how long
    per_device_train_batch_size=4,  # whatever can fit
    per_device_eval_batch_size=4,  # whatever can fit
    logging_strategy='epoch',
    save_strategy='epoch',
    evaluation_strategy='epoch',
    save_total_limit=3,  # keep at most 4 models (one being best model)
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model='rougeLsum',
    output_dir=save_path,
    log_level='critical',
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train_tokenized,
    eval_dataset=ds_val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save adapter
model.save_adapter(save_path, adapter_name)

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
