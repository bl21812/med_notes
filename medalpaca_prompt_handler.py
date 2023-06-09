'''
Prompt handler from Medalpaca
https://github.com/kbressem/medAlpaca/blob/main/medalpaca/handler.py
'''

import json
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def load_json(fn: str):
    with open(fn, "r") as fp:
        d = json.load(fp)
    return d


class DataHandler:
    """Helper class to handle prompt generation and data tokenization.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        prompt_template (str, optional):
            The path to the JSON file containing the prompt template.
            Defaults to "prompts/medalpaca.json".
        model_max_length (int, optional):
            The maximum length of the tokenized sequence.
            Should not exceed 2048, as LLaMA is trained with this. Defaults to 256.
        train_on_inputs (bool, optional):
            If False, masks out inputs in loss. Defaults to True.

    Methods:
        tokenize(prompt: str, add_eos_token: bool = True) -> Dict:
            Tokenizes the given prompt and optionally adds an end-of-sequence (EOS) token.

        generate_and_tokenize_prompt(data_point: Dict) -> Dict:
            Generates a prompt based on the given data point and tokenizes it.

    """

    def __init__(
        self,
        tokenizer,
        prompt_template: str = "prompts/medalpaca.json",
        model_max_length: int = 256,
        train_on_inputs: bool = True,
    ) -> None:
        if model_max_length > 2048:
            logger.warn(f"{model_max_length} exceeds the max token length LLaMA was trained with.")
        self.prompt_template = load_json(prompt_template)
        self.model_max_length = model_max_length
        self.train_on_inputs = train_on_inputs
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str, add_eos_token: bool = True, return_tensors: str = None, truncation: bool = True) -> Dict[str, list]:
        """
        Tokenize the given prompt and optionally add an end-of-sequence (EOS) token.

        This function tokenizes the input prompt without adding special tokens by default.
        If the `add_eos_token` parameter is True and the tokenized sequence doesn't already
        end with an EOS token, an EOS token will be added to the end of the sequence.

        Args:
            prompt (str): The text to be tokenized.
            add_eos_token (bool, optional): Whether to add an EOS token at the end of
                the tokenized sequence. Defaults to True.
            return_tensors (str, optional): If tensors should be returned (and what type).
            trunctaion (bool, optional); Whether to truncate the input to max_model_length
            

        Returns:
            Dict: A dictionary containing the tokenized data:
                - input_ids: The tokenized input IDs of the prompt.
                - attention_mask: The attention mask for the tokenized input IDs.
                - labels: The labels for the tokenized input IDs (identical to input_ids).
        """
        # TODO: optimize (roll back changes from debugging)
        result: Dict = self.tokenizer(
            prompt,
            truncation=truncation,
            max_length=self.model_max_length,
            padding=False,
            return_tensors=return_tensors,
            add_special_tokens=False,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point: Dict):
        """
        Generate a prompt based on the given data point and tokenize it.

        This function creates a prompt using the given data point, which consists
        of an instruction, input, and output. It then tokenizes the generated prompt
        and returns the tokenized representation. If the `train_on_inputs` global
        variable is False, the function will create a user prompt without the
        expected output and only tokenize that part, masking the output part in the
        "labels" field with -100.

        Args:
            data_point (Dict): A dictionary containing the following keys:
                - instruction: The instruction text for the prompt.
                - input: The input text for the prompt.
                - output: The output text for the prompt.

        Returns:
            Dict: A dictionary containing the tokenized prompt and associated data:
                - input_ids: The tokenized input IDs of the generated prompt.
                - attention_mask: The attention mask for the tokenized input IDs.
                - labels: The labels to be used during model training, with the output
                part unmasked and the rest masked with -100 if `train_on_inputs` is False.
        """
        prompt: str = self.generate_prompt(
            instruction=data_point.get("instruction", ""),
            input=data_point.get("input", ""),
            output=data_point.get("output", ""),
        )
        tokenized_prompt: Dict = self.tokenize(prompt)
        if not self.train_on_inputs:
            user_prompt: str = self.generate_prompt(
                instruction=data_point.get("instruction", ""), input=data_point.get("input", "")
            )
            tokenized_user_prompt: Dict = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # mask out the inputs
            tokenized_prompt["labels"] = [
                -100 if i < user_prompt_len else label
                for i, label in enumerate(tokenized_prompt["labels"])
            ]
        return tokenized_prompt

    def generate_prompt(
        self,
        instruction: Optional[str] = None,
        input: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        """
        Generates a prompt for the given instruction, input and output using the specified prompt
        template.

        Args:
            instruction (Optional[str]):
                An optional string representing the instruction to be included in the prompt.
            input (Optional[str]):
                An optional string representing the input to be included in the prompt.
            output (Optional[str]):
                An optional string representing the output to be included in the prompt.

        Returns:
            str: The prompt string created using the specified prompt template.

        Raises:
            ValueError: If none of `instruction`, `input`, and `output` is defined.

        ## Example
        using ``

        {
        "instruction":
        },

        data_handler = DataHandler(tokenizer, "prompt_templates/medalpaca.json")
        prompt = data_hanlder.generate_prompt(
            instruction = "Provide a short answer to this medical question.",
            input = "What to expect if I have Aortic coarctation  (Outlook/Prognosis)?",
            output = (
                "The prognosis of aortic coarctation depends on whether balloon "
                "angioplasty and stenting or the surgery has been done or not."
            )
        )
        print(prompt)
        >>> Below is an instruction that describes a task, paired with an input that provides
            further context. Write a response that appropriately completes the request.

            ### Instruction:
            Provide a short answer to this medical question.

            ### Input:
            What to expect if I have Aortic coarctation  (Outlook/Prognosis)?

            ### Response:
            The prognosis of aortic coarctation depends on whether balloon angioplasty and
            stenting or the surgery has been done or not.
        """

        if not any([instruction, input, output]):
            raise ValueError("At least one of `instruction`, `input`, `output` should be defined")

        prompt = (
            f'{self.prompt_template["primer"]}'
            f'{self.prompt_template["instruction"]}{instruction or ""}'
            f'{self.prompt_template["input"]}{input or ""}'
            f'{self.prompt_template["output"]}{output or ""}'
        )

        return prompt
    
    def generate_prompt_interview(
        self,
        transcript: Optional[str] = None,
        output: Optional[str] = None,
    ):
        
        if not any([transcript, output]):
            raise ValueError("At least one of `transcript` or `output` should be defined")
        
        prompt = (
            '''
            ### Instruction: 

            Given the transcript of a medical patient interview separated by <>, please extract important information. Format your response as JSON with the following structure:

            {
                "History": patient's description of their condition,
                "Physicals": results of any physical examinations, 
                "Diagnosis": doctor's diagnosis, 
                "Plan": doctor's prescribed plan
            }

            To perform this task effectively, follow these steps:
            - First, summarize the entire conversation in bullet points
            - Then, categorize each bullet point based on which of the four sections it belongs to
            '''
            f'{self.prompt_template["transcript"]}{f"<{transcript}>" or ""}'
            f'{self.prompt_template["output"]}{output or ""}'
        )

        return prompt

    def generate_prompt_interview_s_only(self,
        transcript: Optional[str] = None,
        output: Optional[str] = None,
    ):
        
        if not any([transcript, output]):
            raise ValueError("At least one of `transcript` or `output` should be defined")
        
        prompt = (
            f'{self.prompt_template["primer"]}'
            '''
            To perform this task effectively, follow these steps:
            - First, summarize the entire conversation in bullet points
            - Then, summarize the conversation summary with a focus on only the patient's descriptions of the situation

            For example: 

            - The patient presented with a rash on their right leg that has been concerning them for about a week.
            - The patient described the rash as red, with scabs and a larger size.
            - The patient reported scratching at the rash and experiencing no prior occurrences of this type of rash.
            - The patient acknowledged feeling less sensation in their feet over the years.
            '''
            f'{self.prompt_template["transcript"]}{f"<{transcript}>" or ""}'
            f'{self.prompt_template["output"]}{output or ""}'
        )

        return prompt
    
    def generate_prompt_summary(
        self,
        dialogue: Optional[str] = None,
        summary: Optional[str] = None,
    ):
        
        if not any([dialogue, summary]):
            raise ValueError("At least one of `dialogue` or `summary` should be defined")
        
        prompt = (
            f'{self.prompt_template["instruction"]}'
            f'{self.prompt_template["dialogue"]}{dialogue or ""}'
            f'{self.prompt_template["summary"]}{summary or ""}'
        )

        return prompt
    
    def generate_prompt_soap_section(
        self, 
        instruction: Optional[str] = None,
        transcript: Optional[str] = None,
        output: Optional[str] = None,
    ):
        
        prompt = (
            f'{self.prompt_template["instruction"]}{instruction or ""}'
            f'{self.prompt_template["transcript"]}{transcript or ""}'
            f'{self.prompt_template["output"]}{output or ""}'
        )

        return prompt


    def resolve_output(self, output: str): 
        pass


# sample output for one-shot prompting
'''
An example: 

History:
- The patient presented with a rash on their right leg that has been concerning them for about a week.
- The patient described the rash as red, with scabs and a larger size.
- The patient reported scratching at the rash and experiencing no prior occurrences of this type of rash.
- The patient acknowledged feeling less sensation in their feet over the years.

Physicals:
- Physical examination revealed a swollen and red rash on the right ankle, with scabs and signs of an open wound.
- The patient's right leg felt warmer compared to the left.
- The patient experienced pain when walking and flexing the affected leg.
- No loss of feeling was noted specifically in the ankle area.

Diagnosis:
- Possible infected ulcer on the right ankle, exacerbated by poor diabetes management.
- Concerns of cellulitis or another type of infection due to redness, swelling, and pain.
- Potential neuropathy in the feet due to decreased sensation over time.

Plan:
- Order blood tests to assess blood sugar levels and evaluate for infection.
- Recommend the patient to wear compression socks to improve circulation in the area.
- Prescribe appropriate wound care, including cleaning the wound and applying a dressing.
- Follow-up with the patient to discuss test results and adjust the treatment plan accordingly.
'''