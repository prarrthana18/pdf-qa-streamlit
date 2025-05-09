import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import tempfile

st.set_page_config(page_title="PDF QA System", layout="centered")
st.title("📄 PDF Question Answering System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

@st.cache_resource
def load_llm():
    return Ollama(model="llama2")

@st.cache_data
def extract_text_from_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name
    images = convert_from_path(tmp_path)
    text = "".join(pytesseract.image_to_string(img) for img in images)
    os.remove(tmp_path)
    return text

@st.cache_data
def process_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

if uploaded_file:
    st.success("PDF uploaded successfully!")
    with st.spinner("Extracting text and preparing system..."):
        raw_text = extract_text_from_pdf(uploaded_file.read())
        docs = process_text(raw_text)
        embeddings = OllamaEmbeddings(model="llama2")
        vectorstore = Chroma.from_texts(docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=load_llm(), retriever=retriever)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(query)
        st.markdown("### 📌 Answer:")
        st.write(result)
