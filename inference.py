# SCRIPT FOR SAGEMAKER

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer_source = "knkarthick/meeting-summary-samsum"
base_model_source = "knkarthick/meeting-summary-samsum"
adapter_type = 'parallel'

def model_fn(model_dir):

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_source, device_map="auto")
    adapter_name = model.load_adapter(model_dir, config=adapter_type)
    model.set_active_adapters(adapter_name)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")
    
    return {
        'model': model,
        'tokenizer': tokenizer
    }

def transform_fn(model, data, content_type, accept_type):

    # Preprocessing 1. append task indicator
    inp = "summarize: \n\n" + data  

    # Preprocessing 2. use tokenizer (previously loaded into memory) to tokenize input
    tokenized_input = model['tokenizer'](inp, return_tensors="pt")['input_ids']

    # Inference 1. use the model to summarize
    outputs = model['model'].generate(
        tokenized_input,  # uncomment if not using GPU
        # tokenized.to('cuda'),  # uncomment if using GPU
        max_new_tokens=128  # controls max length of summaries 
    )

    # Inference 2. decode to get your final summary as a string
    summary = model['tokenizer'].batch_decode(outputs, skip_special_tokens=True)#[0]

    return summary
