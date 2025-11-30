import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

class TextSummarizer:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def summarize(self, text):
        messages = [
            {"role": "system", "content": "You are a veterinary summarization assistant."},
            {"role": "user", "content": text},
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False)

        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.1
        )

        return self.processor.decode(output[0], skip_special_tokens=True)
