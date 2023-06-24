import torch

from classifier import Classifier

llm_vers = "medalpaca/medalpaca-13b"

classifier = Classifier(llm_vers)

test_input = ['hi guys i am a doctor and this is a renal issue']

output = classifier(test_input)
print(output)

# Load data 

# Train setup
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters())  # lr, weight_decay

# Train loop

