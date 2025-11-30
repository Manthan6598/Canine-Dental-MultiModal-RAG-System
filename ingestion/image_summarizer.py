from PIL import Image

class ImageSummarizer(TextSummarizer):
    def summarize_image(self, image_path):
        img = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": "Describe this veterinary image."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image for dog dental health."}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False)

        inputs = self.processor(
            text=prompt,
            images=[img],
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.1,
        )

        return self.processor.decode(output[0], skip_special_tokens=True)
