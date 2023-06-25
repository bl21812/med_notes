# Class that calls pretrained embedding layer and attention units from an LM
# NOTE: disadvantage here is that my norm layer isn't trainable
    # and it's technically at a different position

import torch
from transformers import AutoModelForCausalLM

class Embedder(torch.nn.Module):

    def __init__(self, llm_vers, num_units=20):
        super().__init__()
        base_lm = AutoModelForCausalLM.from_pretrained(llm_vers, device_map="auto")
        self.embed = base_lm.model.embed_tokens
        self.attention_units = base_lm.model.layers.__getitem__(slice(num_units))
        self.norm = base_lm.model.norm

    def forward(self, x):

        with torch.no_grad():

            embeddings = self.embed(x)

            latents = embeddings
            for unit in self.attention_units:
                if isinstance(latents, tuple):
                    latents = latents[0]
                print(latents.size())
                latents = unit(latents)

            print(latents.size())
            if isinstance(latents, tuple):
                latents = latents[0]
            latents = self.norm(latents)
            print(latents.size())

            latents = [torch.flatten(latents) for l in latents]

        return latents
