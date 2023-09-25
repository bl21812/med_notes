from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer_source = "knkarthick/meeting-summary-samsum"
base_model_source = "knkarthick/meeting-summary-samsum"
adapter_type = 'parallel'

class EndpointHandler():

    def __init__(self, path=""):
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_source, device_map="auto")

        adapter_name = model.load_adapter(path, config=adapter_type)
        model.set_active_adapters(adapter_name)

        model.eval()
        self.model = model

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, device_map="auto")
        self.tokenizer = tokenizer

    def __call__(self, data):
        """
       data args:
            inputs (:obj: `str` | `PIL.Image` | `np.array`)
            kwargs
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """

        # Preprocessing 1. append task indicator
        inp = "summarize: \n\n" + data  

        # Preprocessing 2. use tokenizer (previously loaded into memory) to tokenize input
        tokenized_input = self.tokenizer(inp, return_tensors="pt")['input_ids']

        # Inference 1. use the model to summarize
        outputs = self.model.generate(
            tokenized_input,  # uncomment if not using GPU
            # tokenized.to('cuda'),  # uncomment if using GPU
            max_new_tokens=128  # controls max length of summaries 
        )

        # Inference 2. decode to get your final summary as a string
        summary = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)#[0]

        return summary