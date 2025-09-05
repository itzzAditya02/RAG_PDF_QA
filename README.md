# RAG PDF Q&A with Cerebras

This is a Retrieval-Augmented Generation (RAG) app where you can upload a PDF and ask questions.  
It uses:
- **LangChain** for document processing
- **FAISS** for vector search
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
- **Cerebras LLM** (`llama-3.3-70b`)
- **Streamlit** for UI

## Run Locally
git clone https://github.com/your-username/rag-pdf-qa.git
cd rag-pdf-qa
pip install -r requirements.txt
export CEREBRAS_API_KEY=your_api_key
streamlit run app.py
