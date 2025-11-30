from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import settings
import torch

class Generator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.2,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
