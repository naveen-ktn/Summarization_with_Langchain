from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from langchain import HuggingFaceHub
import os
import tempfile

import os

# Set up HuggingFace API
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

llm = HuggingFaceHub(
    repo_id='mistralai/Mistral-7B-v0.1', model_kwargs={"temperature": 0.5, "max_length": 16384}
)

def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        summaries.append(summary)

        # Delete the temporary file
        os.remove(temp_path)
    
    return summaries

# Streamlit App
st.set_page_config(layout="wide")
st.title("Multiple PDF Summarizer")

# Allow user to upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if pdf_files:
    # Generate summaries when the "Generate Summary" button is clicked
    if st.button("Generate Summary"):
        st.write("Summaries:")
        summaries = summarize_pdfs_from_folder(pdf_files)
        for i, summary in enumerate(summaries):
            st.write(f"Summar.y for PDF {i+1}:")
            st.write(summary)
