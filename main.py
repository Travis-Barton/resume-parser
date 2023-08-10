import openai
import streamlit as st
from utils import file_reader
from llm_class import LLMUtils
from openai import ChatCompletion
from docx import Document
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
fais_db = "vectorstore"
def get_binary_file_downloader_html(bin_file, file_label='File'):
    import base64
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href


# Sidebar for model selection
with st.sidebar:
    model_select = st.selectbox('Select the right model', ['gpt-3.5-turbo-16k', 'gpt-4', 'gpt-3.5-turbo'])
llm_util = LLMUtils(model=ChatCompletion(),
                    embedding_source=FAISS.load_local('vectorstore', embeddings=embeddings),
                    model_names=model_select)

# Main content
st.title("Resume Processor")

# Upload resume
resume_inputer = st.file_uploader('Upload the resume here', type=['doc', 'docx', 'pdf', 'txt'])
if resume_inputer:
    resume = file_reader(resume_inputer)
    sections = ['summary', 'skills_and_tech', 'professional_experience', 'education']
    section_results = {}

    for section in sections:
        result = llm_util.extract_section(resume=resume, section=section)
        section_results[section] = result
        st.write(f"# {section.capitalize().replace('_', ' ')}\n {result}")  # Display extracted content

    # Create a DOCX from the extracted sections
    doc = Document()
    for section, content in section_results.items():
        doc.add_heading(section.capitalize(), level=1)
        doc.add_paragraph(content)
    doc_path = "temp_resume.docx"
    doc.save(doc_path)

    # Allow user to download the consolidated DOCX
    st.markdown(get_binary_file_downloader_html(doc_path, 'Click here to download the consolidated DOCX'), unsafe_allow_html=True)
