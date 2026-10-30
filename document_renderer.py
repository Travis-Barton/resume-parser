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

_SECTION_HEADERS = {
    "summary": {"summary"},
    "skills_and_tech": {"skills_and_tech", "skills and technology", "skills and technologies"},
    "professional_experience": {"professional_experience", "professional experience"},
    "education": {"education"},
}


def content_markup(content, section):
    """Clean the historical per-section response markup before Word styling.

    Word supplies section headings and bullets. Only standalone headings and
    line-leading list markers are removed; prose mentioning those words stays.
    """
    lines = []
    for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"```(?:[ \t]*[A-Za-z0-9_-]+)?", stripped):
            continue
        heading = stripped.strip("#*`: ").casefold()
        if heading in _SECTION_HEADERS.get(section, set()) or stripped == ":":
            continue
        if section == "professional_experience" and heading in {
            "education", "skills", "skills summary", "skills and technology", "skills and technologies"
        }:
            break
        line = line.replace("`", "")
        if section in {"skills_and_tech", "professional_experience"}:
            line = re.sub(r"^[ \t]*(?:\u2022[ \t]*|[*-][ \t]+)", "", line)
        lines.append(line)
    return "\n".join(lines).strip("\r\n")


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
    # The historical prompt uses two job header lines before the bullet text.
    # Preserve blank header slots until after classifying each line's position.
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
        content = content_markup(section_results[key], key)
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
