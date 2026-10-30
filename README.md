# AIM Resume Processor

Convert an uploaded or pasted resume into the existing AIM Word format.

The model picker adds **GPT-6 Luna** and **GPT-6.1 Terra** using the app's original
retrieved-example few-shot method. It searches the existing `vectorstore2`, selects
two relevant examples for each section, and uses the historical section prompts.
Name extraction uses the selected current model too. Legacy picker entries explain
that they are unavailable and direct you to the current choices.

## Run locally

Use Python 3.10 and the existing Streamlit 1.25 series:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Select a model, enter your OpenAI project key, then upload a PDF/DOCX/TXT file or
use **Paste Mode**. Select **Process resume**, review each extracted section, and
use either **Download DOCX** button. Convert older binary `.doc` files to DOCX/PDF
first. Scanned PDFs require text extraction outside this app.

The key entered in this session is used for both example retrieval and extraction;
that project's account is billed. There is no hosted-key override or saved key.
Input/key/model changes clear the previous result. Rerenders and downloads do not
repeat requests. Automatic API retries are disabled.

## Original few-shot method

- The unchanged historical FAISS index uses `text-embedding-ada-002` embeddings.
  Query tokenization, chunk weighting and normalization follow LangChain 0.0.235.
- A single example search replaces the old repeated identical searches. Each
  section still filters its own nonempty targets and takes the top two examples.
- Five Chat Completions calls extract the name, summary, skills, experience and
  education as text. The original formatting prompts and AIM templates are retained.
- Retrieved historical resumes and their formatted sections are sent to OpenAI
  as examples, along with the submitted resume. A fictional submitted resume alone
  does **not** make a request synthetic-only when using the historical index.
- The prompts instruct conservative extraction; review facts and layout before
  sharing. Equivalence to the retired fine-tuned model has not been established.

DOCX files are built in session memory. Raw resumes, prompts, keys and provider
error bodies are not logged. Chat requests set `store=False`; applicable provider
retention policies still govern processing. Truncation, refusal and invalid text
stop export. Input is bounded to 100,000 characters and each section to 8,192
output tokens (name: 128).

## Offline checks

```sh
python -m unittest discover -s tests -v
python -m pip check
```

Tests use fictional examples, temporary synthetic indexes and in-memory HTTP
transports. They do not evaluate live model accuracy or inspect historical records.
The historical index-building script test is skipped: rebuilding/training data is
outside this app stopgap and its old SDK setup needs separate migration.

See [MIGRATION.md](MIGRATION.md) for the source comparison and release checks.
Do not run the historical fine-tuning script casually: its main block starts a
billed job. Neither Luna nor Terra currently supports fine-tuning.

Built with [Streamlit](https://streamlit.io), [OpenAI](https://openai.com), and
[Travis Barton Consulting](https://www.travisbarton.com).
