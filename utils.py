import os
from docx import Document
import PyPDF2

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
        pdf_reader = PyPDF2.PdfFileReader(file_object)
        content = ''
        for page_num in range(pdf_reader.numPages):
            content += pdf_reader.getPage(page_num).extractText()
        return content

    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

# Example usage:
# with open('sample.txt', 'r') as file:
#     print(file_reader(file))
