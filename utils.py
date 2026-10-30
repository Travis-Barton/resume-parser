"""Local text extraction without temporary applicant files or payload logging."""

from io import BytesIO
from pathlib import Path
import re

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pdfminer.high_level import extract_text

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class ResumeFileError(ValueError):
    """Safe, fixed user-facing file error; never carries parser exception text."""


def _read_bytes(file_object):
    try:
        file_object.seek(0)
        content = file_object.read(MAX_UPLOAD_BYTES + 1)
    except Exception:
        raise ResumeFileError("The uploaded file could not be read.") from None
    if isinstance(content, str):
        content = content.encode("utf-8")
    if not isinstance(content, bytes):
        raise ResumeFileError("The uploaded file could not be read.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ResumeFileError("The uploaded file is too large. Use a file smaller than 10 MB.")
    return content


def _docx_text(content):
    document = Document(BytesIO(content))
    parts = []
    seen_headers = set()
    for section in document.sections:
        header = section.header
        if header.part not in seen_headers:
            seen_headers.add(header.part)
            parts.extend(p.text for p in header.paragraphs if p.text.strip())
    # Preserve paragraph/table order; resumes often place names and skills in tables.
    for child in document.element.body:
        if child.tag.endswith('}p'):
            parts.append(Paragraph(child, document).text)
        elif child.tag.endswith('}tbl'):
            table = Table(child, document)
            seen_cells = set()
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    if cell._tc not in seen_cells:
                        seen_cells.add(cell._tc)
                        if cell.text.strip():
                            cells.append(cell.text)
                if cells:
                    parts.append("\t".join(cells))
    return "\n".join(parts)


def file_reader(file_object) -> str:
    """Read DOCX, text-based PDF, or UTF-8 TXT as Unicode text."""
    extension = Path(getattr(file_object, "name", "")).suffix.lower()
    if extension == ".doc":
        raise ResumeFileError("Legacy .doc files are not supported. Save as .docx or PDF and try again.")
    if extension not in {".docx", ".pdf", ".txt"}:
        raise ResumeFileError("Use a DOCX, PDF, or UTF-8 TXT file.")
    content = _read_bytes(file_object)
    try:
        if extension == ".txt":
            return content.decode("utf-8-sig")
        if extension == ".docx":
            return _docx_text(content)
        return extract_text(BytesIO(content))
    except UnicodeDecodeError:
        raise ResumeFileError("The text file is not UTF-8. Save it as UTF-8 and try again.") from None
    except Exception:
        raise ResumeFileError("The file could not be parsed. Try another DOCX, text-based PDF, or UTF-8 TXT file.") from None


def extract_text_from_uploaded_file(uploaded_file):
    """Compatibility wrapper used by earlier callers."""
    return file_reader(uploaded_file)


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
