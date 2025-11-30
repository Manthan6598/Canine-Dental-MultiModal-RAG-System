import gradio as gr
from rag.pipeline import rag_answer

def ui_answer(query):
    answer, img = rag_answer(query)
    return answer, img

with gr.Blocks() as demo:
    gr.Markdown("# 🐶 Canine Dental RAG System (HF CPU Version)")

    q = gr.Textbox(label="Ask a Question")
    a = gr.Textbox(label="Answer")
    i = gr.Image(label="Retrieved Image")

    q.submit(ui_answer, q, [a, i])

demo.launch()
