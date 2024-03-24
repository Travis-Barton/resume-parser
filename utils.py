import os
from docx import Document
from pdfminer.high_level import extract_text
import re
import textract
import tempfile
import io


def file_reader(file_object) -> str:
    """
    A general file reader. doc, docx, pdf, and txt should be taken in and the string should be returned.
    """
    # Extract file extension
    file_ext = os.path.splitext(file_object.name)[1]

    try:
        # Read .txt files
        if file_ext == '.txt':
            content = file_object.read()
            return content

        # Read .docx files
        elif file_ext == '.docx' or file_ext == '.doc':
            doc = Document(file_object)
            fullText = []
            for para in doc.paragraphs:
                fullText.append(para.text)
            return '\n'.join(fullText)

        # Read .pdf files
        elif file_ext == '.pdf':
            with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf') as temp:
                if isinstance(file_object, io.BufferedReader):
                    temp.write(file_object.read())
                else:
                    temp.write(file_object.getvalue())
                temp.flush()
                content = extract_text(temp.name)
            return content

        else:
            raise NotImplementedError(f"Unsupported file extension: {file_ext}")
    except Exception as e:
        # if value error
        if 'application/vnd.openxmlformats-officedocument.themeManager+xm' in str(e):
            raise ValueError('This file is a pre-2007 word-doc. Please convert to a .docx or .pdf file and try again.')
        else:
            raise e

# Example usage:
# with open('sample.txt', 'r') as file:
#     print(file_reader(file))


def extract_text_from_uploaded_file(uploaded_file):
    """
    Extract text from an uploaded file using textract.

    Parameters:
    - uploaded_file: An UploadedFile object from Streamlit's file_uploader

    Returns:
    Extracted text as a string.
    """
    # We'll infer the file extension from the uploaded file's name for textract
    extension = "." + uploaded_file.name.split('.')[-1]

    with tempfile.NamedTemporaryFile(delete=True, suffix=extension) as temp:
        temp.write(uploaded_file.getvalue())
        temp.flush()
        return textract.process(temp.name).decode('utf-8')


def parse_aim_resumes(resume: str):
    """
    AIM resumes have a specific, consistent format.
    """

    # Regular expressions for each section
    section_patterns = {
        'summary': r'SUMMARY\s*(.*?)(?=SKILLS AND TECHNOLOG(?:Y|IES)|PROFESSIONAL EXPERIENCE|EDUCATION|CERTIFICATIONS|AWARDS|$)',
        'skills_and_tech': r'SKILLS AND TECHNOLOG(?:Y|IES)\s*(.*?)(?=PROFESSIONAL EXPERIENCE|EDUCATION|CERTIFICATIONS|AWARDS|$)',
        'professional_experience': r'PROFESSIONAL EXPERIENCE\s*(.*?)(?=EDUCATION|CERTIFICATIONS|AWARDS|$)',
        'education': r'EDUCATION\s*(.*?)(?=CERTIFICATIONS|AWARDS|$)',
        'certifications': r'CERTIFICATIONS\s*(.*?)(?=AWARDS|$)',
        'awards': r'AWARDS\s*(.*)'
    }

    # Extract sections using regex
    sections = {}
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, resume, re.DOTALL)
        sections[section_name] = match.group(1).strip() if match else None

    return (sections['summary'], sections['skills_and_tech'], sections['professional_experience'],
            sections['education'], sections['certifications'], sections['awards'])

if __name__ == '__main__':
    with open('/Users/travisbarton/PycharmProjects/resume_translator/tests/Travis_Barton_Resume.pdf', 'rb') as file:
        print(file_reader(file))
