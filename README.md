A production-grade, modular, multimodal Retrieval-Augmented Generation (RAG) system for veterinary dentistry.

Overview

This project is a multimodal RAG system designed for canine dental health.
It extracts text, tables, and images from veterinary PDFs, summarizes them using a multimodal LLM (Qwen-VL), stores structured embeddings in Pinecone, and finally serves an HF CPU-friendly inference UI using a lightweight model (Mistral / Phi-2).

Key Features

Multimodal ingestion (text + tables + images)
Qwen-VL-7B used offline for image & text summarization
Efficient SentenceTransformer embeddings
Pinecone vector DB for retrieval
Lightweight LLM (Mistral 7B / Phi-2) for HF CPU inference
Gradio UI with retrieved image preview
Production-grade modular code
Dockerized deployment

Architecture:

<img width="485" height="734" alt="image" src="https://github.com/user-attachments/assets/b4160fb3-9960-4e0e-8d12-22b5d5632e8e" />

CANINE MULTIMODAL RAG SYSTEM LINK: https://35dc7ed03105b3bfb4.gradio.live/

Screenshots:

Image1:
<img width="1919" height="1080" alt="image" src="https://github.com/user-attachments/assets/11db3f82-7e67-4b06-9f34-870d957c6c7c" />

Image2:
<img width="1918" height="1068" alt="image" src="https://github.com/user-attachments/assets/e585422b-d500-442e-8a2e-ec2edc403112" />

