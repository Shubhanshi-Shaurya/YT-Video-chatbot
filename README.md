# YT Video Chatbot

An interactive, local Retrieval-Augmented Generation (RAG) web application that allows users to chat seamlessly with any YouTube video. By extracting the transcript, chunking the text, and storing vector representations locally, users can ask multiple follow-up questions about the video's content using a conversational chat interface.

##  Features
* **Zero API Key Metadata Retrieval:** Automatically scrapes the YouTube video title and pulls the high-resolution thumbnail using lightweight oEmbed API configurations.
* **Local & Secure:** Powered by **Ollama (Llama 3)** and local multilingual embeddings, keeping your processing data private.
* **Persistent Chat Experience:** Utilizes Streamlit's session state to maintain a smooth, multi-turn conversation history.
* **Multilingual Capabilities:** Uses a robust multilingual embedding model to interpret and translate context gracefully if the query language differs from the video's spoken language.

---

## Tech Stack

* **Frontend UI:** [Streamlit](https://streamlit.io/)
* **Orchestration Framework:** [LangChain](https://www.langchain.com/) (using LangChain Expression Language - LCEL)
* **Vector Database:** [FAISS-CPU](https://github.com/facebookresearch/faiss)
* **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via Hugging Face
* **Local LLM:** [Ollama](https://ollama.com/) (`llama3`)
* **Data Extraction:** `youtube-transcript-api`

---

##  Getting Started

### 1. Prerequisites
Make sure you have [Ollama](https://ollama.com/) installed on your machine and have the Llama 3 model pulled locally:
```bash
ollama pull llama3
```

### 2. Installation
Clone this repository and navigate to the project directory:

git clone [https://github.com/Shubhanshi-Shaurya/YT-Video-chatbot](https://github.com/Shubhanshi-Shaurya/YT-Video-chatbot)
``` bash
cd YT-Video-chatbot

```

Create a virtual environment and activate it:
```bash 
python -m venv .venv
# On Windows:
.venv\Scripts\activate
```

Install the required dependencies:
```bash 
pip install -r requirements.txt
```

### 3. Running the App
Launch the Streamlit interface:
```bash 
streamlit run app.py
```

---