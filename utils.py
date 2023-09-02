import os
from docx import Document
from pdfminer.high_level import extract_text
import re

def file_reader(file_object) -> str:
    """
    A general file reader. doc, docx, pdf, and txt should be taken in and the string should be returned.
    """

    # Extract file extension
    file_ext = os.path.splitext(file_object.name)[1]

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
        content = extract_text(file_object)
        return content

    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

# Example usage:
# with open('sample.txt', 'r') as file:
#     print(file_reader(file))

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