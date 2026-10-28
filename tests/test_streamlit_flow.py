"""Streamlit interaction tests with a mocked provider and synthetic input only."""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

from docx import Document
from streamlit.testing.v1 import AppTest

from llm_class import MODEL_OPTIONS, ResumeProcessingError


APP = Path(__file__).resolve().parents[1] / "main.py"
SYNTHETIC_RESUME = "Synthetic Alex Rivera. Example Co engineer, 2020–2024. Skills: Python and SQL."
SECTIONS = {
    "name": "Synthetic Alex Rivera",
    "summary": "Engineer at Example Co.",
    "skills_and_tech": "Python\nSQL",
    "professional_experience": "Example Co\t2020–2024\nEngineer\nBuilt synthetic examples",
    "education": "",
}


def prepared_app():
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    app.checkbox[0].check().run()
    app.text_input[0].input("SYNTHETIC_KEY").run()
    app.text_area[0].input(SYNTHETIC_RESUME).run()
    return app


class StreamlitFlowTests(unittest.TestCase):
    def test_initial_page_and_missing_input_never_initialize_provider(self):
        with patch("llm_class.LLMUtils") as llm:
            app = AppTest.from_file(str(APP), default_timeout=10).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button[0].disabled)
            app.text_input[0].input("SYNTHETIC_KEY").run()
            self.assertTrue(app.button[0].disabled)
            llm.assert_not_called()

    def test_ready_inputs_wait_for_explicit_process(self):
        with patch("llm_class.LLMUtils") as llm:
            app = prepared_app()
            self.assertFalse(app.exception)
            self.assertFalse(app.button[0].disabled)
            llm.assert_not_called()
            app.run()
            llm.assert_not_called()

    def test_resume_without_key_keeps_processing_disabled(self):
        with patch("llm_class.LLMUtils") as llm:
            app = AppTest.from_file(str(APP), default_timeout=10).run()
            app.checkbox[0].check().run()
            app.text_area[0].input(SYNTHETIC_RESUME).run()
            self.assertTrue(app.button[0].disabled)
            llm.assert_not_called()

    def test_process_uses_entered_key_and_creates_download(self):
        with patch("llm_class.LLMUtils") as llm:
            llm.return_value.extract_resume.return_value = SECTIONS
            app = prepared_app()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            llm.assert_called_once_with(api_key="SYNTHETIC_KEY", model_name=next(iter(MODEL_OPTIONS.values())))
            llm.return_value.extract_resume.assert_called_once_with(SYNTHETIC_RESUME)
            self.assertEqual(len(app.get("download_button")), 1)
            doc = Document(BytesIO(app.session_state["resume_docx"]))
            self.assertIn("Synthetic Alex Rivera", "\n".join(p.text for p in doc.paragraphs))
            app.run()
            llm.assert_called_once()

    def test_changed_resume_clears_stale_download_without_new_call(self):
        with patch("llm_class.LLMUtils") as llm:
            llm.return_value.extract_resume.return_value = SECTIONS
            app = prepared_app()
            app.button[0].click().run()
            app.text_area[0].input("Different synthetic applicant").run()
            self.assertFalse(app.get("download_button"))
            self.assertNotIn("resume_docx", app.session_state)
            llm.assert_called_once()

    def test_changed_model_clears_stale_download(self):
        with patch("llm_class.LLMUtils") as llm:
            llm.return_value.extract_resume.return_value = SECTIONS
            app = prepared_app()
            app.button[0].click().run()
            app.selectbox[0].select(list(MODEL_OPTIONS)[1]).run()
            self.assertFalse(app.get("download_button"))
            llm.assert_called_once()

    def test_changed_key_clears_stale_download(self):
        with patch("llm_class.LLMUtils") as llm:
            llm.return_value.extract_resume.return_value = SECTIONS
            app = prepared_app()
            app.button[0].click().run()
            app.text_input[0].input("DIFFERENT_SYNTHETIC_KEY").run()
            self.assertFalse(app.get("download_button"))
            llm.assert_called_once()

    def test_failed_repeat_processing_does_not_keep_previous_download(self):
        with patch("llm_class.LLMUtils") as llm:
            llm.return_value.extract_resume.side_effect = [SECTIONS, RuntimeError("SENTINEL_PRIVATE_BODY")]
            app = prepared_app()
            app.button[0].click().run()
            self.assertEqual(len(app.get("download_button")), 1)
            app.button[0].click().run()
            self.assertFalse(app.get("download_button"))
            self.assertNotIn("resume_docx", app.session_state)
            self.assertNotIn("SENTINEL", app.error[0].value)

    def test_safe_provider_error_stops_before_export(self):
        with patch("llm_class.LLMUtils") as llm:
            llm.return_value.extract_resume.side_effect = ResumeProcessingError(
                "authentication", "OpenAI could not authenticate this API key. Check the selected key."
            )
            app = prepared_app()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("could not authenticate", app.error[0].value)
            self.assertFalse(app.get("download_button"))

    def test_unknown_error_does_not_expose_provider_content(self):
        with patch("llm_class.LLMUtils") as llm:
            llm.return_value.extract_resume.side_effect = RuntimeError("SENTINEL_PRIVATE_PROVIDER_BODY")
            app = prepared_app()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("SENTINEL", app.error[0].value)
            self.assertFalse(app.get("download_button"))


if __name__ == "__main__":
    unittest.main()
