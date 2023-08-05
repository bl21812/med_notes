import torch

from transformers.adapters import ParallelConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

tokenizer_source = "knkarthick/meeting-summary-samsum"
base_model_source = "knkarthick/meeting-summary-samsum"

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

# believe i can use this instead of AutoAdapterModel ?
model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_source, 
    device_map='auto'
)

# idk if parallel adapter is good for few shot
'''config = ParallelConfig(
    mh_adapter=True,
    output_adapter=True,  # can keep both of these in for now (unsure if needed)
    reduction_factor=16,  # important param !! (not sure what val)
    non_linearity="relu"
)
model.add_adapter("bottleneck_adapter", config=config)

model.train_adapter("bottleneck_adapter")
model.set_active_adapters("bottleneck_adapter")
print_trainable_parameters(model)'''

tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")

example = """
D: OK. Your lymph nodes don’t feel swollen to me, which is a good sign. So here’s what I think should be our next steps. I’m going to order an exercise stress test for you to do, which will help figure out if there’s anything wrong with your heart. I’m also going to order bloodwork to rule out any possible infection that might be causing your chest pain. In the meantime, I’m going to prescribe you 2 pills of aspirin to take as needed when you feel that chest pain, and we’ll see if that helps relieve the pain. And let’s follow up once we get all the test results back. My office will contact you to set up an appointment in a few weeks. How does that sound? 

P: Sounds great Doc. Thanks!
"""

tokenized = tokenizer(example, return_tensors='pt')['input_ids']
# print(tokenized)

# decoded = tokenizer.decode(tokenized)
# print(decoded)

generation_config = GenerationConfig(
        max_new_tokens=64,
        temperature=0.1,
        top_p=0.75,
        top_k=40,  # higher = more memory
        num_beams=1,  # higher = more memory
        # early_stopping=True, 
        # no_repeat_ngram_size=3  # need to take into account summary contexts! (what is the longest sequence that could repeat)
    )

with torch.no_grad():
    outputs = model.generate(
        tokenized.to('cuda'),
        generation_config=generation_config
    )  # input shouldnt be a list ??

print(outputs)
print(tokenizer.batch_decode(outputs)[0])
