# Current-model stopgap

## Preserved historical method

Baseline `bfb642b` (master) uses the old FAISS retrieval method whenever the picker
selects a non-fine-tuned model. `find_similar_resumes` retrieves up to 100 ranked
records from `vectorstore2`. `format_resume_examples` filters records with a target
for the requested section and uses the first two. Commit `9d7b438` deliberately
reduced the number of examples from three to two. The `e00fb62` change added a
separate name extraction call.

The revised stopgap preserves those section prompts and example wrappers, text
outputs, template files, section order, upload/paste interface and two downloads.
Both current model choices use that original method. It does not use the initial
PR's fictional static examples, single JSON extraction request or JSON adapter.

`index_resumes.py` originally used LangChain 0.0.235's default OpenAIEmbeddings:
`text-embedding-ada-002`, `cl100k_base`, 8,191-token chunks, token-weighted average
and L2 normalization. The index bytes remain unchanged. A small adapter applies
that same query embedding procedure through the current client, avoiding the old
embedding class's incompatible SDK calls and retries. The same search is cached
within one processing attempt instead of repeating it four times. Each content
section still chooses its own top two relevant examples.

## Necessary compatibility and safety changes

- Add `gpt-6-luna` and `gpt-6.1-terra` to the existing picker. Legacy entries give
  an unavailable-model message before any requests. The known failing default
  is `ft:gpt-3.5-turbo-0613:personal::7vbb2i7t`, explicitly rejected as deprecated.
- Use Chat Completions with the original prompts, `reasoning_effort="none"`,
  `max_completion_tokens` and `store=False`. Name extraction uses the chosen model.
- Use OpenAI 3.41.0 per-client credentials and `max_retries=0` for both embeddings
  and chat. The old SDK has automatic transport/chat retry behavior.
- Keep Streamlit 1.25.0 and Python 3.10 compatibility; no Cloud runtime upgrade is
  required. The dev app's traceback showed Python 3.10. `.python-version` documents
  local setup and does not change an existing Cloud runtime.
- Use only the entered key; remove the hidden hosted-secret override that also
  crashed dev when its secret was absent. Both API routes bill the entered project.
- Remove raw-resume/prompt logs and temporary prompt files. Expose sanitized errors.
- Require an explicit Process action, clear stale results and generate DOCX in
  session memory. These avoid repeat charges and shared applicant files.
- Keep the old output cleanup for section text, bullets and fences before applying
  the existing template styles. File reading handles DOCX/PDF/TXT without shared
  temporary files. Legacy binary `.doc` is explicitly rejected.

## Evidence and release checks

The dependency set resolves for Linux CPython 3.10. On Windows Python 3.10.20,
61 unittest cases ran: 60 passed and the historical index-builder test was skipped.
The actual Streamlit 1.25 ScriptRunner opened the initial app without credentials,
then completed paste, explicit processing, both downloads and rerender with a mocked
extractor and a valid fictional DOCX. All 89 installed packages passed compatibility
checks. A fictional two-role DOCX was rendered in LibreOffice and visually inspected
without clipping, overlapping text or duplicate bullets. This is known-answer
renderer evidence, not model-output quality evidence. Browser interaction, Linux
execution, historical index loading and live model inference remain unverified.

Offline checks cover original prompt/example selection, one search per resume,
selected-model name extraction, client isolation, zero retries, sanitized errors,
synthetic FAISS persistence/search, UI workflow, and DOCX formatting. They cannot
establish current project access, live output quality or latency.

A complete historical-method processing attempt uses one embedding request and
five chat requests. Do not apply the previous single-request smoke calculation.
The historical examples can contain real applicant data: synthetic-only live tests
must inject a fictional example source instead of loading production `vectorstore2`.
Any live test uses the parent's remaining aggregate approved budget, never a new
allowance. No live calls are implied by the offline test commands.

Before release, confirm the actual Cloud repository/branch/entrypoint, compare
current PR/base, then merge the exact tested head under existing deployment
authorization. Inspect Cloud build/startup and the current-model picker. Report
live processing as unverified until a real authorized attempt succeeds and its
DOCX is inspected. The old master is not a working model rollback.

## Fine-tuning follow-up

Luna and Terra currently do not support fine-tuning. Eligible existing organizations
may have a short-lived SFT bridge using older GPT-4.1/mini bases, but organization
eligibility is unverified. New organizations have been excluded since May 7, 2026;
from July 2, organizations lacking fine-tuned inference in the preceding 60 days
are excluded. Remaining job creation ends January 6, 2027, and the relevant older
models shut down January 14, 2027. This is not a durable restoration route.

Any job requires separately authorized, appropriately permitted/redacted training
examples, a held-out evaluation, account eligibility and a current cost estimate.
The API minimum is ten examples; guidance recommends about fifty high-quality
demonstrations. Historical scripts/artifacts are leads, not authorization to load
customer records or start training. No training job has been launched.

## Sources

- [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)
- [Current API parameters](https://developers.openai.com/api/docs/guides/latest-model)
- [Original embedding implementation](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/embeddings/openai.py)
- [Original FAISS loader](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/vectorstores/faiss.py)
- [Deprecations](https://developers.openai.com/api/docs/deprecations)
- [Supervised fine-tuning](https://developers.openai.com/api/docs/guides/supervised-fine-tuning)
