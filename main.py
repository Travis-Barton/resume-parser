"""Streamlit entry point. Provider calls happen only after Process resume."""

import streamlit as st

from document_renderer import SECTION_ORDER, render_resume_docx
from llm_class import LLMUtils, MODEL_OPTIONS, ResumeProcessingError
from utils import ResumeFileError, file_reader


def clear_result():
    for key in ("resume_sections", "resume_docx"):
        st.session_state.pop(key, None)


st.title("AIM Resume Processor")
st.caption(
    "Luna and Terra use example-guided extraction. These are new model options, "
    "not the retired fine-tuned models. Review the extracted facts before sharing."
)

with st.sidebar:
    model_label = st.selectbox("Model", list(MODEL_OPTIONS), on_change=clear_result)
    api_key = st.text_input(
        "OpenAI API key", type="password", on_change=clear_result,
        help="This session uses the key you enter. Your OpenAI project is billed for processing.",
    )
    st.caption("Processing starts only when you select Process resume.")

paste_mode = st.checkbox("Paste Mode", value=False, on_change=clear_result)
if paste_mode:
    source = st.text_area("Paste the resume here", on_change=clear_result)
else:
    source = st.file_uploader(
        "Upload the resume here", type=["docx", "pdf", "txt"], on_change=clear_result,
        help="For an older .doc file, save it as .docx or PDF first.",
    )

if not api_key.strip():
    st.info("Enter your OpenAI API key in the sidebar to process a resume.")

if st.button("Process resume", disabled=not api_key.strip() or not source):
    clear_result()
    try:
        resume = source if paste_mode else file_reader(source)
    except ResumeFileError as exc:
        st.error(str(exc))  # ResumeFileError contains only fixed, safe messages.
        st.stop()
    except Exception:
        st.error("The file could not be read. Try a DOCX, text-based PDF, or UTF-8 text file.")
        st.stop()

    if not resume.strip():
        st.error("No readable text was found. For a scanned PDF, paste its text instead.")
        st.stop()

    try:
        with st.spinner("Extracting the resume and preparing the document..."):
            sections = LLMUtils(
                api_key=api_key.strip(), model_name=MODEL_OPTIONS[model_label]
            ).extract_resume(resume)
            document_bytes = render_resume_docx(sections)
    except ResumeProcessingError as exc:
        st.error(str(exc))  # The client exposes sanitized categories, not provider payloads.
        st.stop()
    except Exception:
        st.error("The document could not be prepared. No file was exported. Please try again.")
        st.stop()

    st.session_state["resume_sections"] = sections
    st.session_state["resume_docx"] = document_bytes

if "resume_sections" in st.session_state:
    sections = st.session_state["resume_sections"]
    if not sections["name"].strip():
        st.warning("No applicant name was found. Check the source before sharing the document.")
    for section in ("name", *SECTION_ORDER):
        with st.expander(f"Extracted {section.replace('_', ' ').upper()}"):
            st.text(sections[section] or "No information provided in the source.")
    st.download_button(
        label="Download DOCX", data=st.session_state["resume_docx"],
        file_name="consolidated.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    st.button("Clear result", on_click=clear_result)

with st.sidebar:
    st.markdown("---")
    st.markdown("[Report an issue](https://github.com/Travis-Barton/resume-parser/issues)")
    st.markdown("_Made by [Travis Barton Consulting](https://www.travisbarton.com)_")
