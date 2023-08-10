import openai
import streamlit as st
from utils import file_reader
from llm_class import LLMUtils
from openai import ChatCompletion
from docx import Document
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


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
    api_key = st.text_input('Enter your OpenAI API key', type='password')
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        embeddings = OpenAIEmbeddings()
        llm_util = LLMUtils(model=ChatCompletion(),
                            embedding_source=FAISS.load_local('vectorstore', embeddings=embeddings),
                            model_names=model_select)

# Main content
st.title("AIM Resume Processor")

# Upload resume
resume_inputer = st.file_uploader('Upload the resume here', type=['doc', 'docx', 'pdf', 'txt'])
if api_key:
    if resume_inputer:
        resume = file_reader(resume_inputer)
        sections = ['summary', 'skills_and_tech', 'professional_experience', 'education']
        section_results = {}

        for section in sections:
            result = llm_util.extract_section(resume=resume, section=section)
            section_results[section] = result
            st.write(f"# {section.replace('_', ' ').capitalize()}\n {result}")  # Display extracted content

        # Create a DOCX from the extracted sections
        doc = Document()
        for section, content in section_results.items():
            doc.add_heading(section.capitalize(), level=1)
            doc.add_paragraph(content)
        doc_path = "AIM Profile.docx"
        doc.save(doc_path)

        # Allow user to download the consolidated DOCX
        with st.sidebar:
            st.markdown(get_binary_file_downloader_html(doc_path, 'Click here to download the consolidated DOCX'),
                        unsafe_allow_html=True)
        st.markdown(get_binary_file_downloader_html(doc_path, 'Click here to download the consolidated DOCX'),
                    unsafe_allow_html=True)
else:
    st.markdown('__Please enter your OpenAI API key in the sidebar__')

with st.sidebar:
    st.markdown('---')
    st.markdown("<br><br><br><br>Something broken? <br>[File an issue](https://github.com/Travis-Barton/resume-parser/issues) or "
                "reach out to me <a href='mailto:me@travisbarton.com?subject=Resume Parser'>by email</a>",
                unsafe_allow_html=True)
    st.markdown('<br>_Made by [Travis Barton Consulting](https://www.travisbarton.com)_',
                unsafe_allow_html=True)
