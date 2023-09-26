import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel

iam_client = boto3.client('iam')
sess = sagemaker.Session()

role = 'arn:aws:iam::912547607495:role/service-role/AmazonSageMaker-ExecutionRole-20230925T211953'

# Hub model configuration <https://huggingface.co/models>
hub = {
  'HF_MODEL_ID':'bl21812/summ_beta0_adapter', # model_id from hf.co/models
  'HF_TASK':'custom'                           # NLP task you want to use for predictions
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub,                                                # configuration for loading model from Hub
   role=role,                                              # IAM role with permissions to create an endpoint
   transformers_version="4.30.2",                             # Transformers version used
   pytorch_version="2.01",                                  # PyTorch version used
   py_version='py39',                                      # Python version used
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.m4.xlarge",
   endpoint_name='quip-ai-summ-beta0'
)
