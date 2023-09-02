import openai
import streamlit as st
from utils import file_reader
from llm_class import LLMUtils
from openai import ChatCompletion
from docx import Document
from docx.shared import Pt
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import dotenv
fais_db = "vectorstore2"

replace_dict = {
    "summary": "SUMMARY_REPLACE",
    "skills_and_tech": "SKILL_REPLACE",
    "professional_experience": "JOBEXPERIENCE_REPLACE",
    "education": "EDUCATION_REPLACE",
    "certifications": "CERTIFICATION_REPLACE",
    "awards": "AWARD_REPLACE"
}



def replace_text_in_paragraph(paragraph, key, value):
    if key in paragraph.text:
        inline = paragraph.runs
        for item in inline:
            if key in item.text:
                item.text = item.text.replace(key, value)


def get_binary_file_downloader_html(bin_file, file_label='File'):
    import base64
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href


# Sidebar for model selection
with st.sidebar:
    model_select = st.selectbox('Select the right model', ['gpt-3.5-turbo-16k', 'gpt-4', 'gpt-3.5-turbo', 'fine-tuned-gpt-3.5-turbo-4k'])
    api_key = st.text_input('Enter your OpenAI API key', type='password')
    if api_key:
        openai.api_key = api_key
        embeddings = OpenAIEmbeddings()
        llm_util = LLMUtils(model=ChatCompletion(),
                            embedding_source=FAISS.load_local(fais_db, embeddings=embeddings),
                            model_names=model_select)
    if model_select == 'fine-tuned-gpt-3.5-turbo-4k':
        openai.api_key = st.secrets['OPENAI_API_KEY']

# Main content
st.title("AIM Resume Processor")

# Upload resume
resume_inputer = st.file_uploader('Upload the resume here', type=['doc', 'docx', 'pdf', 'txt'])
if api_key:
    st.success('Press R to rerun the analysis')
    if resume_inputer:
        resume = file_reader(resume_inputer)
        sections = ['summary', 'skills_and_tech', 'professional_experience', 'education', 'certifications', 'awards']
        section_results = {}

        for section in sections:
            result = llm_util.extract_section(resume=resume, section=section)
            section_results[section] = result
            st.markdown(f"# {section.replace('_', ' ').upper()}\n {result}",
                        unsafe_allow_html=True)  # Display extracted content
        # Create a DOCX from the extracted sections
        # read doucment
        # doc = Document('AIM Profile template.docx')
        doc = Document('AIM Profile - Template.docx')
        for section, content in section_results.items():
            # style = doc.styles['Normal']
            # font = style.font
            # font.name = 'Calibri (Body)'
            # font.size = Pt(11)
            # doc.add_heading(section.replace('_', ' ').upper().replace('SKILLS AND TECH', 'SKILLS AND TECHNOLOGY'), level=1)
            # style = doc.styles['Normal']
            # font.name = 'Calibri (Body)'
            content = content.replace("\n- ", "\n• ")
            content = content.replace("\t- ", "\t• ")
            content = content.replace("\nSUMMARY\n", "")
            content = content.replace("\nSKILLS AND TECHNOLOGY\n", "")
            content = content.replace("\nSKILLS_AND_TECH\n", "")
            content = content.replace("\nPROFESSIONAL EXPERIENCE\n", "")
            content = content.replace("\nEDUCATION\n", "")
            content = content.replace('```education', '')
            content = content.replace("`", "").replace('skills_and_tech', '')
            content = content.replace('professional_experience', '')
            content = content.replace('- ', '• ')
            if section == 'professional_experience':
                content = content.split('\nEducation')[0]
            if content.startswith(':\n'):
                content = content[2:]
            if content.startswith('education'):
                content = content[9:]
            for paragraph in doc.paragraphs:
                replace_text_in_paragraph(paragraph, replace_dict[section], content.strip())
            # doc.add_paragraph(content)
            # style = doc.styles['Normal']
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
    st.markdown(
        "<br><br><br><br>Something broken? <br>[File an issue](https://github.com/Travis-Barton/resume-parser/issues) or "
        "reach out to me <a href='mailto:me@travisbarton.com?subject=Resume Parser'>by email</a>",
        unsafe_allow_html=True)
    st.markdown('<br>_Made by [Travis Barton Consulting](https://www.travisbarton.com)_',
                unsafe_allow_html=True)
