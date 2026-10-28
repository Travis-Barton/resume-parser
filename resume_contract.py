"""Validate extracted resume data and adapt it to the existing DOCX sections.

This module is pure and requires only the standard library. It does not read
files, use credentials, or make API calls. Blank experience header slots are
represented by a tab so that ``\n\n`` remains an unambiguous job separator.
Renderers must preserve these slots when selecting paragraph styles.
"""

import unicodedata


SECTION_KEYS = (
    "name", "summary", "skills_and_tech", "professional_experience", "education"
)
JOB_FIELDS = ("employer", "location", "title", "start_date", "end_date", "bullets")
MAX_TOTAL_TEXT_CHARS = 100_000
MAX_JOBS = 100
MAX_BULLETS_PER_JOB = 100
MAX_SKILLS = 500
MAX_EDUCATION_ITEMS = 100
MAX_NAME_CHARS = 500
MAX_HEADER_CHARS = 1_000
MAX_DATE_CHARS = 100
MAX_ITEM_CHARS = 5_000


def _string_schema(max_length):
    return {"type": "string", "maxLength": max_length}


def _array_schema(max_items):
    return {
        "type": "array",
        "items": _string_schema(MAX_ITEM_CHARS),
        "maxItems": max_items,
    }


RESUME_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(SECTION_KEYS),
    "properties": {
        "name": _string_schema(MAX_NAME_CHARS),
        "summary": _string_schema(MAX_TOTAL_TEXT_CHARS),
        "skills_and_tech": _array_schema(MAX_SKILLS),
        "professional_experience": {
            "type": "array",
            "maxItems": MAX_JOBS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(JOB_FIELDS),
                "properties": {
                    "employer": _string_schema(MAX_HEADER_CHARS),
                    "location": _string_schema(MAX_HEADER_CHARS),
                    "title": _string_schema(MAX_HEADER_CHARS),
                    "start_date": _string_schema(MAX_DATE_CHARS),
                    "end_date": _string_schema(MAX_DATE_CHARS),
                    "bullets": _array_schema(MAX_BULLETS_PER_JOB),
                },
            },
        },
        "education": _array_schema(MAX_EDUCATION_ITEMS),
    },
}


class ResumeValidationError(ValueError):
    """Extraction did not meet the contract; messages contain no input values."""


def _require_fields(value, fields, path):
    if type(value) is not dict:
        raise ResumeValidationError(f"{path} must be an object.")
    if set(value) != set(fields):
        raise ResumeValidationError(f"{path} has missing or unexpected fields.")


def _require_array(value, maximum, path):
    if type(value) is not list:
        raise ResumeValidationError(f"{path} must be an array.")
    if len(value) > maximum:
        raise ResumeValidationError(f"{path} has too many items.")


def validate_resume(data):
    """Return a fresh normalized object, or raise ``ResumeValidationError``.

    Missing facts must be represented by empty strings/arrays, not absent keys
    or nulls. Only the summary may contain line breaks. Total original text is
    limited to 100,000 characters in addition to the per-field/item bounds.
    """
    _require_fields(data, SECTION_KEYS, "resume")
    total_chars = 0

    def text(value, maximum, path, multiline=False):
        nonlocal total_chars
        if type(value) is not str:
            raise ResumeValidationError(f"{path} must be a string.")
        if len(value) > maximum:
            raise ResumeValidationError(f"{path} is too long.")
        total_chars += len(value)
        if total_chars > MAX_TOTAL_TEXT_CHARS:
            raise ResumeValidationError("resume exceeds the total text limit.")
        if multiline:
            value = value.replace("\r\n", "\n").replace("\r", "\n")
        for char in value:
            codepoint = ord(char)
            xml_character = (
                codepoint in (0x9, 0xA, 0xD)
                or 0x20 <= codepoint <= 0xD7FF
                or 0xE000 <= codepoint <= 0xFFFD
                or 0x10000 <= codepoint <= 0x10FFFF
            )
            if not xml_character:
                raise ResumeValidationError(f"{path} contains a character unsupported by the document format.")
            if multiline and char == "\n":
                continue
            if unicodedata.category(char) == "Cc" or char in "\u2028\u2029":
                raise ResumeValidationError(f"{path} contains unsupported control characters or line breaks.")
        return value.strip()

    def items(value, maximum, path):
        _require_array(value, maximum, path)
        normalized = []
        for index, item in enumerate(value):
            item = text(item, MAX_ITEM_CHARS, f"{path}[{index}]")
            if item:
                normalized.append(item)
        return normalized

    result = {
        "name": text(data["name"], MAX_NAME_CHARS, "resume.name"),
        "summary": text(data["summary"], MAX_TOTAL_TEXT_CHARS, "resume.summary", multiline=True),
        "skills_and_tech": items(data["skills_and_tech"], MAX_SKILLS, "resume.skills_and_tech"),
        "professional_experience": [],
        "education": [],
    }
    jobs = data["professional_experience"]
    _require_array(jobs, MAX_JOBS, "resume.professional_experience")
    for index, job in enumerate(jobs):
        path = f"resume.professional_experience[{index}]"
        _require_fields(job, JOB_FIELDS, path)
        result["professional_experience"].append({
            "employer": text(job["employer"], MAX_HEADER_CHARS, path + ".employer"),
            "location": text(job["location"], MAX_HEADER_CHARS, path + ".location"),
            "title": text(job["title"], MAX_HEADER_CHARS, path + ".title"),
            "start_date": text(job["start_date"], MAX_DATE_CHARS, path + ".start_date"),
            "end_date": text(job["end_date"], MAX_DATE_CHARS, path + ".end_date"),
            "bullets": items(job["bullets"], MAX_BULLETS_PER_JOB, path + ".bullets"),
        })
    result["education"] = items(data["education"], MAX_EDUCATION_ITEMS, "resume.education")
    return result


def _plain_bullet(value):
    # The Word paragraph style supplies the visual bullet. Preserve punctuation
    # inside the content, including .NET, Node.js, dates, and negative numbers.
    if value.startswith("•"):
        return value[1:].lstrip()
    if value.startswith(("- ", "* ")):
        return value[2:].lstrip()
    return value


def to_section_results(data):
    """Return exactly the five legacy renderer keys with string values."""
    resume = validate_resume(data)
    skills = [_plain_bullet(item) for item in resume["skills_and_tech"]]
    blocks = []
    for job in resume["professional_experience"]:
        bullets = [_plain_bullet(item) for item in job["bullets"]]
        bullets = [item for item in bullets if item]
        employer_line = ", ".join(value for value in (job["employer"], job["location"]) if value)
        date_span = " - ".join(value for value in (job["start_date"], job["end_date"]) if value)
        first_header = "\t".join(value for value in (employer_line, date_span) if value)
        if not first_header and not job["title"] and not bullets:
            continue
        lines = [first_header or "\t", job["title"] or "\t", *bullets]
        blocks.append("\n".join(lines))
    return {
        "name": resume["name"],
        "summary": resume["summary"],
        "skills_and_tech": "\n".join(item for item in skills if item),
        "professional_experience": "\n\n".join(blocks),
        "education": "\n".join(resume["education"]),
    }
