# AIM Resume Processor

Convert a PDF, DOCX, UTF-8 text file, or pasted resume into the existing AIM Word format.

The app now offers **GPT-6 Luna** and **GPT-6.1 Terra** with synthetic few-shot examples
and structured extraction. The retired GPT-3.5 fine-tunes are no longer used.
These are replacement extraction modes; equivalent quality to the former fine-tune
has not been established.

## Run locally

Use Python 3.12:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Select a model, enter your OpenAI project API key into the password field, and
upload or paste a resume. Click **Process resume**, review the extracted facts,
and download the DOCX.

The entered key is used for this session's client. There is no hidden hosted-key
override and no key stored in source or an environment file by this app.
Your selected OpenAI project is billed when you process. Editing the input, key,
or model clears the previous result. Downloads do not call the model again.
For old binary `.doc` files, convert to DOCX/PDF first; scanned PDFs require
text extraction/OCR outside this app.

## Format and data handling

- Output: applicant name, summary, skills/technologies, professional experience,
  and education, using the existing AIM templates and paragraph styles.
- Missing facts remain blank. The app does not invent contract status or dates.
- Structured JSON is validated and converted to document sections in Python.
- DOCX output is generated in memory per session, not a shared `resume.docx`.
- Raw resumes, prompts, API keys, and provider error bodies are not logged.
- Responses are requested with `store=False`. This is not a claim of zero provider
  retention; applicable API data policies still govern processing.
- Input is capped at 100,000 characters and output at 8,192 tokens. Refused,
  truncated, or invalid model output does not produce a partial document.

## Offline tests

```sh
python -m unittest discover -s tests -v
python -m pip check
```

Tests use synthetic fixtures and mocked/in-memory transports. They verify code
and document behavior, not real-model extraction quality. The historical FAISS
indexing test is explicitly skipped: retrieval is not part of the current app
and those scripts need their old dependencies and authorized data.

See [MIGRATION.md](MIGRATION.md) for capability sources, required live quality
checks, and current fine-tuning limitations. Do not run the historical training
script casually: its main block can start a billed fine-tuning job.

Built with [Streamlit](https://streamlit.io), [OpenAI](https://openai.com), and
[Travis Barton Consulting](https://www.travisbarton.com).

