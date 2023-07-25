from transformers.adapters import ParallelConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

input()
