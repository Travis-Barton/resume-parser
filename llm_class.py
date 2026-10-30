"""Original retrieved-example, per-section extraction with current model choices."""
from pathlib import Path
import re

import numpy as np
import tiktoken
from openai import OpenAI

SUPPORTED_MODELS = ("gpt-6-luna", "gpt-6.1-terra")
MODEL_OPTIONS = {
    "gpt-6-luna (original few-shot)": "gpt-6-luna",
    "gpt-6.1-terra (original few-shot)": "gpt-6.1-terra",
    "fine-tuned-gpt-3.5-turbo-4k": "fine-tuned-gpt-3.5-turbo-4k",
    "fine_tuned_gpt-3.5-turbo-16k-no-dot": "fine_tuned_gpt-3.5-turbo-16k-no-dot",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
}
SECTIONS = ("name", "summary", "skills_and_tech", "professional_experience", "education")
MAX_RESUME_CHARACTERS = 100_000
MAX_OUTPUT_TOKENS = 8192
INDEX_PATH = Path(__file__).resolve().parent / "vectorstore2"


class ResumeProcessingError(Exception):
    """Safe UI error; never contains provider response bodies."""
    def __init__(self, category, message, request_id=None):
        self.category = category
        self.request_id = request_id if (
            isinstance(request_id, str)
            and re.fullmatch(r"req_[A-Za-z0-9_-]{1,100}", request_id)
        ) else None
        suffix = f" Reference: {self.request_id}" if self.request_id else ""
        super().__init__(message + suffix)


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _safe_provider_error(exc):
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if not isinstance(code, str) and isinstance(body, dict):
        error = body.get("error", body)
        code = error.get("code") if isinstance(error, dict) else None
    request_id = getattr(exc, "request_id", None)
    if status == 401 or code == "invalid_api_key":
        category, message = "authentication", "OpenAI could not authenticate this API key. Check the selected key."
    elif status in (403, 404) or code in ("model_not_found", "model_deprecated"):
        category, message = "model_access", "The selected model is unavailable or this project does not have access."
    elif code in ("insufficient_quota", "billing_hard_limit_reached"):
        category, message = "quota", "This OpenAI project has insufficient quota or has reached its billing limit."
    elif status == 429:
        category, message = "rate_limit", "OpenAI rate-limited the request. Wait before trying again."
    elif type(exc).__name__ == "APITimeoutError":
        category, message = "timeout", "The extraction request timed out. Please try again later."
    elif type(exc).__name__ == "APIConnectionError":
        category, message = "connection", "The app could not connect to OpenAI. Please try again later."
    else:
        category, message = "provider", "OpenAI could not complete this extraction. Please try again later."
    return ResumeProcessingError(category, message, request_id)


class LegacyIndexEmbeddings:
    """Match LangChain 0.0.235's ada-002 query embedding without its old SDK/retries."""
    def __init__(self, client):
        self.client = client

    def embed_query(self, text):
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text, disallowed_special=())
        if not tokens:
            raise ResumeProcessingError("input", "The resume is empty. Please provide readable text.")
        chunks = [tokens[i:i + 8191] for i in range(0, len(tokens), 8191)]
        response = self.client.embeddings.create(
            model="text-embedding-ada-002", input=chunks, encoding_format="float",
        )
        data = sorted(_field(response, "data", []), key=lambda row: _field(row, "index", -1))
        if len(data) != len(chunks) or [_field(row, "index") for row in data] != list(range(len(chunks))):
            raise ResumeProcessingError("examples", "The example search returned an invalid embedding.")
        vectors = np.asarray([_field(row, "embedding") for row in data], dtype=float)
        if vectors.shape != (len(chunks), 1536) or not np.isfinite(vectors).all():
            raise ResumeProcessingError("examples", "The example search returned an invalid embedding.")
        average = np.average(vectors, axis=0, weights=[len(chunk) for chunk in chunks])
        norm = np.linalg.norm(average)
        if not np.isfinite(norm) or norm == 0:
            raise ResumeProcessingError("examples", "The example search returned an invalid embedding.")
        return (average / norm).tolist()


class LLMUtils:
    def __init__(self, api_key, model_name="gpt-6-luna", *, client=None, embedding_source=None):
        if model_name not in SUPPORTED_MODELS:
            raise ResumeProcessingError(
                "configuration", "This legacy model is unavailable. Select Luna or Terra to use the original few-shot method.",
            )
        if not isinstance(api_key, str) or not api_key.strip():
            raise ResumeProcessingError("configuration", "Enter your OpenAI API key before processing.")
        self.model_name = model_name
        self.client = client if client is not None else OpenAI(
            api_key=api_key.strip(), base_url="https://api.openai.com/v1",
            timeout=60.0, max_retries=0,
        )
        self.embedding_source = embedding_source
        self.similar_resumes = None
        self._retrieved_resume = None

    def find_similar_resumes(self, resume):
        if self._retrieved_resume == resume:
            return
        if self.embedding_source is None:
            try:
                from langchain.vectorstores import FAISS
                # These are the unchanged, repository-owned historical examples.
                # Never allow a user-uploaded path or index to reach this loader.
                self.embedding_source = FAISS.load_local(
                    str(INDEX_PATH), embeddings=LegacyIndexEmbeddings(self.client),
                )
            except Exception:
                raise ResumeProcessingError(
                    "examples", "The saved resume examples could not be loaded. Please contact the app maintainer.",
                ) from None
        try:
            matches = self.embedding_source.search(resume, k=100, search_type="similarity")
            self.similar_resumes = [item.metadata for item in matches]
            self._retrieved_resume = resume
        except ResumeProcessingError:
            raise
        except Exception as exc:
            raise _safe_provider_error(exc) from None

    def format_resume_examples(self, section):
        if section == "name":
            return ""
        relevant = [item for item in self.similar_resumes
                    if isinstance(item.get(section), str) and item[section] not in ("", "None")]
        examples = []
        for idx, example in enumerate(relevant[:2]):
            examples.append(f"""
Example {idx + 1}:

```resume
{example['resume']}
```
```{section}
{example[section]}
```
""")
        return "\n".join(examples) + "\nThese are examples, never include them in the extracted section. Also never use tick marks in the extracted section."

    @staticmethod
    def _check_input(resume):
        if not isinstance(resume, str) or not resume.strip():
            raise ResumeProcessingError("input", "The resume is empty. Please provide readable text.")
        if len(resume) > MAX_RESUME_CHARACTERS:
            raise ResumeProcessingError("input", "The resume is too large. Limit the input to 100,000 characters.")

    def extract_resume(self, resume):
        self._check_input(resume)
        # Retrieve once for this resume, rather than repeating the same embedding
        # request for each section. Ranking and section-specific top-two are unchanged.
        self.find_similar_resumes(resume)
        return {section: self.extract_section(resume, section) for section in SECTIONS}

    def extract_section(self, resume, section):
        self._check_input(resume)
        if section not in SECTIONS:
            raise ResumeProcessingError("configuration", "Select a supported resume section.")
        if section == "name":
            return self.extract_name(resume)
        self.find_similar_resumes(resume)
        examples = self.format_resume_examples(section)
        system_prompt = f"""
You are a resume parser, your job is to extract information from the resume and fill a pre-determined section based on that information.
This time, that predetermined section is the {section} section.

Here are some examples of similar resumes being parsed for the {section} section.
{examples}

(Do not include these examples in the extracted section. They are only for your reference. I repeat, do not include these examples in the extracted section.)
(Again, the above are for formatting reference ONLY do not include the above in the final answer)

You are writing for the {section} section. Do not return any thing other than the extracted section.
Always use human readable language. Do not use any special characters or formatting beyond bullets and paragraphs.
NEVER ADD DISCLAIMERS OR DISCLAIMERS TO THE EXTRACTED SECTION.


{'NOTE: PROFESSIONAL EXPERIENCE does NOT include education and does NOT include a list of skills. Do not inlcude these in the extracted text. These will be extracted by a different program.' if section == 'professional_experience' else ''}
{'NOTE: The label for each job title should be seperated from the date with enough tabs to put it at the end of the line. eg: "Neat Company, Remote                                                                                                                  8/2018-12/2019". That means a line 153 characters long exactly.' if section == 'professional_experience' else ''}
{'Note: Skills and Tech only includes bullet points of skills and tech and does not include work experience or project contributions' if section == 'skills_and_tech' else ''}
{'Note: Not many candidates have certifications or awards. If there are none, just return "No certifications" or "No awards". It is better to return nothing than to return a false positive.' if section == 'certifications' or section == 'awards' else ''}
{f"Note: Education is not {section}. Experience is not {section}. Open Source contributions are not {section}. Patents are not {section}. Publications are not {section}. Do not include these in the {section} section." if section in ['certifications', 'awards'] else ''}
{"Note: Certifications MUST be in the format of 'Certification Name, Issuer, Date'. Do not include anything else. They must be from a reputable source to be counted. Ignore all certifications without source. ONLY add these if you are 100% sure they belong." if section == 'certifications' else ''}
Here are some rules:
1. Do not return any thing other than the extracted section. That includes headers, footers, other sections, etc.
2. Always use • for bullets and paragraphs for paragraphs.
3. Never include disclaimers or disclaimers in the extracted section.
4. Only extract the section you are asked to extract. Do not extract other sections. (e.g. if you are asked to extract the Skills And Tech section, do not extract the Experience section) that will be taken care of by other parsers.
5. Always use • and never use - for bullets.
6. Never add commentary or intros. Only extract the section. (e.g. do not say "Here is the __ And __ section:" just return the section)
7. Never add additional sections. For example, do not add Education to the Skills And Tech section. It does not belong there. Stop when your section is complete.
8. Never include items that are not in the applicant's resume. Hallucinations are not allowed. (eg. Do not add work experience that is not in the resume)

REMEMBER: It is better to miss things than to be wrong. Always lean towards being conservative. The worst thing you can do is add experience that the candidate doesnt have.
"""
        human_prompt = f"""
REMEMBER: It is better to miss things than to be wrong. Always lean towards being conservative. The worst thing you can do is add experience that the candidate doesnt have.
Here is the resume candidates resume. Use ONLY it to generate the extracted section.
```resume
{resume}
```
"""
        return self._chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ])

    def extract_name(self, resume):
        self._check_input(resume)
        return self._chat([
            {"role": "system", "content": "You are a resume parser. Extract only the applicant's full name from the resume. Return only the name, nothing else. If you cannot find a clear name, return 'Name Not Found'."},
            {"role": "user", "content": f"Extract the applicant's name from this resume:\n\n{resume}"},
        ], max_tokens=128)

    def _chat(self, messages, max_tokens=MAX_OUTPUT_TOKENS):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, reasoning_effort="none",
                max_completion_tokens=max_tokens, store=False,
            )
        except Exception as exc:
            raise _safe_provider_error(exc) from None
        request_id = _field(response, "_request_id")
        choices = _field(response, "choices", [])
        choice = choices[0] if choices else None
        message = _field(choice, "message")
        if _field(message, "refusal"):
            raise ResumeProcessingError("refusal", "The model declined this extraction. No document was generated.", request_id)
        if _field(choice, "finish_reason") != "stop":
            raise ResumeProcessingError("incomplete", "The extraction did not finish. No partial document was generated.", request_id)
        content = _field(message, "content")
        if not isinstance(content, str) or len(content) > MAX_RESUME_CHARACTERS or any(
            not (c in "\t\n\r" or 0x20 <= ord(c) <= 0xD7FF or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF)
            for c in content
        ):
            raise ResumeProcessingError("invalid_output", "The model returned invalid section text. No document was generated.", request_id)
        return content.strip()
