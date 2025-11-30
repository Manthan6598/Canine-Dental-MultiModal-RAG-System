import os
import re
import torch
import gradio as gr
from PIL import Image
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForVision2Seq

# ========== Load Models ==========
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ========== Pinecone Setup ==========
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set as an environment variable.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("multimodal-rag-qwen")

# ========== Cleaning Helpers ==========

def clean_qwen_output(text: str) -> str:
    text = re.sub(r"system.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"user.*?\n", "", text, flags=re.IGNORECASE)
    text = text.replace("[image]", "")
    text = text.replace("addCriterion", "")
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def extract_answer(full_output: str) -> str:
    if "Question:" in full_output:
        full_output = full_output.split("Question:")[-1]
    full_output = re.sub(r"^.*?\?", "", full_output, count=1).strip()
    return clean_qwen_output(full_output)

def extract_top_image(results):
    for m in results["matches"]:
        if m["metadata"].get("type") == "image":
            return {
                "path": m["metadata"].get("image_path"),
                "description": clean_qwen_output(m["metadata"].get("summary", ""))
            }
    return None

# ========== RAG Core ==========

def rag_answer(query: str):
    # 1. Embed query
    q_emb = embed_model.encode(query).tolist()

    # 2. Retrieve from Pinecone
    results = index.query(
        vector=q_emb,
        top_k=5,
        include_metadata=True
    )

    # 3. Best image (if any)
    top_image = extract_top_image(results)

    # 4. Build context from summaries
    context = ""
    for m in results["matches"]:
        t = m["metadata"].get("type")
        summary = clean_qwen_output(m["metadata"].get("summary", ""))

        if t in ["text", "table"]:
            context += summary + "\n"
        elif t == "image":
            context += "Image Description: " + summary + "\n"

    # 5. Ask Qwen for final answer
    messages = [
        {"role": "system", "content": "You are a veterinary expert."},
        {"role": "user", "content": context + f"\nQuestion: {query}"}
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.1
    )

    answer_raw = processor.decode(output[0], skip_special_tokens=True)
    final_answer = extract_answer(answer_raw)

    return final_answer, top_image

# ========== Gradio UI ==========

def ui_answer(query: str):
    if not query.strip():
        return "Please enter a question.", None, ""

    answer, top_image = rag_answer(query)

    img = None
    desc = ""

    if top_image and top_image["path"]:
        try:
            # NOTE: In your Pinecone index, image_path currently points to Colab paths.
            # On HuggingFace this may fail; in that case, we just show description.
            img = Image.open(top_image["path"])
        except Exception:
            img = None

        desc = top_image["description"]

    return answer, img, desc


with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center;'>Canine Dental RAG System</h1>")
    gr.Markdown(
        "<p style='text-align:center;'>Ask questions about canine dental health. The system retrieves text + the most relevant image description.</p>"
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Ask a Question",
                placeholder="What is gingivitis in dogs?",
                lines=3,
            )

    with gr.Row():
        with gr.Column(scale=2):
            answer_box = gr.Textbox(
                label="Final Answer",
                lines=10,
            )
        with gr.Column(scale=1):
            image_box = gr.Image(
                label="Relevant Image",
                height=350,
                width=350
            )
            desc_box = gr.Textbox(
                label="Image Description",
                lines=6,
            )

    submit_btn = gr.Button("Submit", variant="primary")

    submit_btn.click(
        ui_answer,
        inputs=question_box,
        outputs=[answer_box, image_box, desc_box]
    )

if __name__ == "__main__":
    demo.launch()
