# Building LLM Chatbots: From Fundamentals to RAG

This repository contains the complete code and presentation materials for the **"Building Chatbots with LLMs"** workshop, hosted by **Women in AI, Ireland** and sponsored by **EY**, held on **15 October 2025**.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4)](https://ai.google.dev/gemini-api)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-f34a17)](https://www.langchain.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00)](https://www.gradio.app/)
[![Hugging Face Embeddings Model & Image Generation](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/)

## Workshop Overview

This workshop is divided into two hands-on labs designed to provide a complete learning journey:

1.  **Lab 1: LLM Fundamentals & Multimodal Chatbot**
    * You'll start by understanding the core concepts of LLMs, such as prompts, parameters, and streaming. We'll use the **Google Gemini API** to make our first API calls and build an interactive **Creative Travel Planner** that can generate text, images, and audio.
2.  **Lab 2: Retrieval-Augmented Generation (RAG)**
    * Building on the fundamentals, you'll learn how to overcome the limitations of standard LLMs (like knowledge cutoffs). We'll use **LangChain** to build a RAG pipeline that connects an LLM to a custom knowledge base of recent EU AI Act news, creating a Q&A application called the **EU AI News Navigator**.

---

## Repository Structure

```bash
├── Lab1_Basic_Chatbot.ipynb           # Lab 1 notebook
├── Lab2_RAG.ipynb                     # Lab 2 notebook
├── EU_AI_Act_Corpus                   # Mini dataset used in Lab 2 (text sources for RAG)
├── WAIWorkshop-Building_Chatbots.pdf  # Theory slides covering LLMs and RAG concepts
├── setup_guide.pdf                    # Step-by-step Gemini API setup guide
├── sample_output                      # Screenshots of sample outputs
└── README.md
```
---

## Getting Started

The easiest way to run these labs is by using Google Colab, which provides a free, cloud-based Python environment.

### Prerequisites

* A Google Account (for Google Colab).
* Python 3.9+ (if running locally).
* **API Keys**: You will need two free API keys to run the notebooks fully.
    1.  **Google Gemini API Key**: Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
    2.  **Hugging Face User Access Token**: Required for image generation in Lab 1. Get one from your [Hugging Face Settings](https://huggingface.co/settings/tokens).

### Running on Google Colab (Recommended)

Click the buttons below to open each notebook directly in Google Colab.

[![Open Lab 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomaIjatomi/WAIWorkshop-Building-Chatbots/blob/main/Lab1_Basic_Chatbot.ipynb)

[![Open Lab 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomaIjatomi/WAIWorkshop-Building-Chatbots/blob/main/Lab2_RAG.ipynb)
  
---

## What You'll Learn in Each Lab

### Lab 1: LLM Fundamentals & a Multimodal Chatbot

This lab focuses on the foundational skills for interacting with LLMs.

**Key Concepts Covered:**
* **LLM Parameters**: `temperature`, `top_p`, and `max_output_tokens`.
* **System Prompts**: How to give your LLM a persona or specific instructions.
* **Streaming**: Generating token-by-token responses for a real-time feel.
* **Conversation Memory**: Implementing simple, short-term memory for chatbots.
* **Gradio UI**: Building an interactive web interface for your model.
* **Multimodality**: Integrating text generation (**Gemini**), image generation (**Stable Diffusion**), and audio synthesis (**gTTS**).

**Final Project: The Creative Travel Planner**
An app that generates personalized travel itineraries, complete with inspiring images and a spoken summary.

![Creative Travel Planner App](https://github.com/TomaIjatomi/WAIWorkshop-Building-Chatbots/blob/main/Sample_Output/Creative%20Travel%20Planner%20-%20Sample%20Output.png)

### Lab 2: Retrieval-Augmented Generation (RAG)

This lab addresses a critical enterprise use case: making LLMs reason over your own private or recent data.

**Key Concepts Covered:**
* **The RAG Pipeline**: Understanding the flow from loading data to generating answers. 
* **Document Loading**: Ingesting external documents (`.txt` files).
* **Chunking (Splitting)**: Breaking down large documents into manageable pieces. 
* **Embeddings & Vector Stores**: Converting text to vectors for semantic search using **FAISS**. 
* **LangChain Chains**: Composing the `retriever`, `prompt`, and `LLM` into a seamless RAG chain.

**Final Project: EU AI News Navigator**
A Q&A app that accurately answers questions about recent EU AI Act news, citing its sources to build user trust.

![RAG Demo](https://github.com/TomaIjatomi/WAIWorkshop-Building-Chatbots/blob/main/Sample_Output/RAG%20Demo1%20-%20EU%20AI%20Act%20Recent%20News.png)

---

## Key Technologies Used

| Technology       | Description                                                                     |
| :--------------- | :------------------------------------------------------------------------------ |
| **Google Gemini** | The powerful, multimodal large language model used for text generation. The Lab utilizes Gemini 2.0 Flash model       |
| **LangChain** | The framework used to orchestrate the RAG pipeline components.                 |
| **Hugging Face** | Provider of the embedding model (`all-MiniLM-L6-v2`) and image generation API.  |
| **Gradio** | A simple and fast way to build and share interactive UIs for machine learning models. |
| **FAISS** | An efficient similarity search library used as our in-memory vector store.        |

## Acknowledgements

**Facilitators:**
  - [Toma Ijatomi](https://www.linkedin.com/in/toma-ijatomi/)
  - [Shweta Soni](https://www.linkedin.com/in/shweta-soni-7a9a2012a/)

This session was organized by Women in AI Ireland in collaboration with EY under the WAI Education initiative.

Delivered in person on 15 October 2025 at EY Dublin.
www.womeninai.co

<img src="https://github.com/TomaIjatomi/WAIWorkshop-Building-Chatbots/blob/main/workshop_flier.png" alt="WAI Workshop Flier" width="400"/>
---

