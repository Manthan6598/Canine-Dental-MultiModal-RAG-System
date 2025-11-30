import re
from rag.retriever import Retriever
from rag.generator import Generator

retriever = Retriever()
generator = Generator()

def clean(text):
    t = re.sub(r"\[image.*?\]", "", text)
    return t.strip()

def build_context(results):
    ctx = ""
    for m in results["matches"]:
        ctx += clean(m["metadata"]["summary"]) + "\n"
    return ctx

def rag_answer(query):
    results = retriever.search(query)
    context = build_context(results)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    answer = generator.generate(prompt)

    # pick best image
    image = None
    for m in results["matches"]:
        if m["metadata"]["type"] == "image":
            image = m["metadata"]["image_path"]
            break

    return answer, image
