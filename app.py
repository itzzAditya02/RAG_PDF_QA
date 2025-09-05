import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# ======================
# Config
# ======================
load_dotenv()
MODEL_NAME = "llama-3.3-70b"
client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

def ask_cerebras(prompt: str) -> str:
    try:
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=2000,
            temperature=0.3
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"⚠️ Error during generation: {str(e)}"

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="📑 RAG PDF Q&A", layout="wide")

st.title("📑 RAG PDF Q&A with Cerebras")
st.write("Upload a PDF and ask questions. Powered by LangChain + FAISS + Cerebras LLM.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Embeddings + FAISS
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding)
    retriever = db.as_retriever()

    st.success("✅ PDF processed! You can now ask questions.")

    # Query box
    query = st.text_input("Enter your question:")

    if query:
        relevant_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        You are a helpful assistant. Use the following context to answer the question.
        Context: {context}
        Question: {query}
        Answer:
        """

        with st.spinner("Generating answer..."):
            answer = ask_cerebras(prompt)

        st.markdown(f"### 🤖 Answer\n{answer}")

        # Show context toggle
        with st.expander("📂 Show Retrieved Context"):
            st.write(context)

