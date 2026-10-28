"""Render validated section text into the existing AIM Word templates in memory."""

from io import BytesIO
from pathlib import Path
import re

from docx import Document
from docxcompose.composer import Composer


SECTION_TEMPLATES = {
    "summary": "SUMMARY.docx",
    "skills_and_tech": "SKILLS AND TECHNOLOGIES.docx",
    "professional_experience": "PROFESSIONAL EXPERIENCE.docx",
    "education": "EDUCATION.docx",
}
SECTION_ORDER = tuple(SECTION_TEMPLATES)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _paragraphs(container):
    yield from container.paragraphs
    for table in container.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from _paragraphs(cell)


def replace_text_in_paragraph(paragraph, key, value):
    """Replace a template token even if Word split it across styled runs."""
    matches = list(re.finditer(re.escape(key), paragraph.text))
    for match in reversed(matches):
        position = 0
        inserted = False
        for run in paragraph.runs:
            text = run.text
            run_end = position + len(text)
            if position < match.end() and run_end > match.start():
                prefix = text[:max(0, match.start() - position)]
                suffix = text[max(0, match.end() - position):]
                run.text = prefix + (value if not inserted else "") + suffix
                inserted = True
            position = run_end


def format_professional_experience(content, document):
    # The contract reserves two header slots per job. An absent header is a tab,
    # so preserve whitespace until after classifying the line's position.
    for job in content.strip("\r\n").split("\n\n"):
        if not job.strip():
            continue
        for index, line in enumerate(job.split("\n")):
            if not line.strip():
                continue
            style = "Travis Normal Bold 2" if index < 2 else "List Bullet 2"
            document.add_paragraph(line, style=style)


def format_skills_and_tech(content, document):
    for line in content.splitlines():
        if line.strip():
            document.add_paragraph(line, style="List Bullet 2")


def render_resume_docx(section_results, template_dir=None):
    """Return a DOCX byte string without writing shared applicant files."""
    expected = {"name", *SECTION_ORDER}
    if set(section_results) != expected or any(
        not isinstance(section_results[key], str) for key in expected
    ):
        raise ValueError("The extracted resume has an invalid section structure.")

    templates = Path(template_dir) if template_dir is not None else TEMPLATE_DIR
    master = Document(templates / "AIM Profile - Template.docx")
    containers = [master]
    for section in master.sections:
        containers.extend([section.header, section.footer])
    for container in containers:
        for paragraph in _paragraphs(container):
            replace_text_in_paragraph(paragraph, "APPLICANT_NAME", section_results["name"].strip())

    composer = Composer(master)
    for key, filename in SECTION_TEMPLATES.items():
        document = Document(templates / filename)
        content = section_results[key]
        if key == "professional_experience":
            format_professional_experience(content, document)
        elif key == "skills_and_tech":
            format_skills_and_tech(content, document)
        elif content.strip():
            document.add_paragraph(content, style="Travis Normal")
        composer.append(document)

    composed = BytesIO()
    composer.save(composed)
    composed.seek(0)
    final_document = Document(composed)
    containers = [final_document]
    for section in final_document.sections:
        containers.extend([section.header, section.footer])
    for container in containers:
        for paragraph in _paragraphs(container):
            for run in paragraph.runs:
                run.font.name = "Calibri"
    output = BytesIO()
    final_document.save(output)
    return output.getvalue()
