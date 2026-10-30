"""UI control-flow tests using only APIs available in Streamlit 1.25.

The fake executes the real main.py and models widget callbacks before reruns.
It does not establish browser rendering or claim to run the real Streamlit runtime.
No AppTest import, real credentials, retrieval store, or provider calls are used.
"""

from io import BytesIO
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock

from docx import Document

APP = Path(__file__).resolve().parents[1] / "main.py"
MODEL_OPTIONS = {
    "gpt-6-luna (original few-shot)": "gpt-6-luna",
    "gpt-6.1-terra (original few-shot)": "gpt-6.1-terra",
    "fine-tuned-gpt-3.5-turbo-4k": "fine-tuned-gpt-3.5-turbo-4k",
    "fine_tuned_gpt-3.5-turbo-16k-no-dot": "fine_tuned_gpt-3.5-turbo-16k-no-dot",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
}
SYNTHETIC_RESUME = "Synthetic Alex Rivera. Example Co engineer, 2020-2024. Skills: Python and SQL."
SECTIONS = {
    "name": "Synthetic Alex Rivera",
    "summary": "Engineer at Example Co.",
    "skills_and_tech": "• Python\n• SQL",
    "professional_experience": "Example Co\t2020-2024\nEngineer\n• Built synthetic examples",
    "education": "",
}
KEY_LABEL = "Enter your OpenAI API key"
MODEL_LABEL = "Select the right model"
TEXT_LABEL = "Paste the resume here"


class StopExecution(Exception):
    pass


class SafeProcessingError(Exception):
    pass


class Scope:
    def __init__(self, app, location):
        self.app = app
        self.location = location

    def __enter__(self):
        self.previous = self.app.location
        self.app.location = self.location
        return self

    def __exit__(self, *args):
        self.app.location = self.previous


class TrackedSecrets(dict):
    def __init__(self):
        super().__init__(OPENAI_API_KEY="SYNTHETIC_HOSTED_KEY")
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)


class FakeStreamlit(ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.secrets = TrackedSecrets()
        self.values = {}
        self.callbacks = {}
        self.calls = []
        self.location = "main"
        self.sidebar = Scope(self, "sidebar")
        self.pressed = set()

    def record(self, kind, value=None, **kwargs):
        self.calls.append({"kind": kind, "value": value, "location": self.location, **kwargs})

    def calls_of(self, kind):
        return [item for item in self.calls if item["kind"] == kind]

    def widget(self, kind, label, default, **kwargs):
        self.record(kind, label, **kwargs)
        self.callbacks[label] = kwargs.get("on_change")
        return self.values.setdefault(label, default)

    def selectbox(self, label, options, **kwargs):
        return self.widget("selectbox", label, options[0], options=list(options), **kwargs)

    def text_input(self, label, **kwargs):
        return self.widget("text_input", label, "", **kwargs)

    def checkbox(self, label, value=False, **kwargs):
        return self.widget("checkbox", label, value, **kwargs)

    def text_area(self, label, **kwargs):
        return self.widget("text_area", label, "", **kwargs)

    def file_uploader(self, label, **kwargs):
        return self.widget("file_uploader", label, None, **kwargs)

    def button(self, label, disabled=False, on_click=None, **kwargs):
        self.record("button", label, disabled=disabled, **kwargs)
        self.callbacks[label] = on_click
        was_pressed = label in self.pressed
        self.pressed.discard(label)
        return was_pressed and not disabled

    def download_button(self, label, **kwargs):
        self.record("download_button", label, **kwargs)
        return False

    def expander(self, label, **kwargs):
        self.record("expander", label, **kwargs)
        return Scope(self, self.location)

    def spinner(self, label):
        self.record("spinner", label)
        return Scope(self, self.location)

    def markdown(self, text, **kwargs):
        self.record("markdown", text, **kwargs)

    def text(self, text):
        self.record("text", text)

    def caption(self, text):
        self.record("caption", text)

    def title(self, text):
        self.record("title", text)

    def warning(self, text):
        self.record("warning", text)

    def error(self, text):
        self.record("error", text)

    def stop(self):
        raise StopExecution()

    def change(self, label, value):
        self.values[label] = value
        callback = self.callbacks.get(label)
        if callback:
            callback()

    def click(self, label):
        callback = self.callbacks.get(label)
        if callback:
            callback()
        self.pressed.add(label)

    def run(self, provider):
        self.calls = []
        llm_module = ModuleType("llm_class")
        llm_module.MODEL_OPTIONS = MODEL_OPTIONS
        llm_module.LLMUtils = provider
        llm_module.ResumeProcessingError = SafeProcessingError
        # Restore only the two mocked entries. Replacing all of sys.modules
        # between reruns unloads newly imported native modules; some Python
        # 3.10 extensions (including cryptography) cannot initialize twice.
        missing = object()
        overrides = {"streamlit": self, "llm_class": llm_module}
        previous = {name: sys.modules.get(name, missing) for name in overrides}
        sys.modules.update(overrides)
        try:
            try:
                runpy.run_path(str(APP), run_name="__main__")
            except StopExecution:
                pass
        finally:
            for name, module in previous.items():
                if module is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return self


def prepared_app(provider):
    app = FakeStreamlit().run(provider)
    app.change("Paste Mode", True)
    app.run(provider)
    app.change(KEY_LABEL, "SYNTHETIC_KEY")
    app.change(TEXT_LABEL, SYNTHETIC_RESUME)
    return app.run(provider)


class StreamlitFlowTests(unittest.TestCase):
    def provider(self):
        provider = Mock()
        provider.return_value.extract_resume.return_value = dict(SECTIONS)
        return provider

    def test_original_labels_upload_and_branding_are_present(self):
        provider = self.provider()
        app = FakeStreamlit().run(provider)
        self.assertEqual(app.calls_of("title")[0]["value"], "AIM Resume Processor")
        self.assertEqual(app.calls_of("selectbox")[0]["value"], MODEL_LABEL)
        self.assertEqual(app.calls_of("selectbox")[0]["options"], list(MODEL_OPTIONS))
        self.assertEqual(app.calls_of("text_input")[0]["value"], KEY_LABEL)
        self.assertEqual(app.calls_of("file_uploader")[0]["value"], "Upload the resume here")
        self.assertEqual(app.calls_of("checkbox")[0]["value"], "Paste Mode")
        footers = "\n".join(item["value"] for item in app.calls_of("markdown"))
        for text in ("Whats new?", "File an issue", "mailto:me@travisbarton.com", "Travis Barton Consulting"):
            self.assertIn(text, footers)
        provider.assert_not_called()

    def test_initial_page_and_missing_input_never_initialize_provider(self):
        provider = self.provider()
        app = FakeStreamlit().run(provider)
        self.assertTrue(app.calls_of("button")[0]["disabled"])
        app.change(KEY_LABEL, "SYNTHETIC_KEY")
        app.run(provider)
        self.assertTrue(app.calls_of("button")[0]["disabled"])
        provider.assert_not_called()

    def test_ready_inputs_wait_for_explicit_process(self):
        provider = self.provider()
        app = prepared_app(provider)
        self.assertFalse(app.calls_of("button")[0]["disabled"])
        app.run(provider)
        provider.assert_not_called()

    def test_resume_without_key_keeps_processing_disabled(self):
        provider = self.provider()
        app = prepared_app(provider)
        app.change(KEY_LABEL, "")
        app.click("Process resume")
        app.run(provider)
        self.assertTrue(app.calls_of("button")[0]["disabled"])
        provider.assert_not_called()

    def test_process_uses_entered_key_and_both_download_locations(self):
        provider = self.provider()
        app = prepared_app(provider)
        app.click("Process resume")
        app.run(provider)
        provider.assert_called_once_with(api_key="SYNTHETIC_KEY", model_name="gpt-6-luna")
        provider.return_value.extract_resume.assert_called_once_with(SYNTHETIC_RESUME)
        downloads = app.calls_of("download_button")
        self.assertEqual([item["location"] for item in downloads], ["sidebar", "main"])
        self.assertEqual(downloads[0]["data"], downloads[1]["data"])
        self.assertEqual(downloads[1]["key"], "download_button2")
        self.assertEqual(downloads[0]["file_name"], "consolidated.docx")
        doc = Document(BytesIO(downloads[0]["data"]))
        self.assertIn("Synthetic Alex Rivera", "\n".join(p.text for p in doc.paragraphs))
        self.assertEqual([item["value"] for item in app.calls_of("expander")], [
            "Extracted NAME", "Extracted SUMMARY", "Extracted SKILLS AND TECH",
            "Extracted PROFESSIONAL EXPERIENCE", "Extracted EDUCATION",
        ])
        app.run(provider)
        provider.assert_called_once()

    def test_hosted_secret_cannot_override_entered_session_key(self):
        for label, model in list(MODEL_OPTIONS.items())[:2]:
            with self.subTest(model=model):
                provider = self.provider()
                app = prepared_app(provider)
                app.change(MODEL_LABEL, label)
                app.click("Process resume")
                app.run(provider)
                provider.assert_called_once_with(api_key="SYNTHETIC_KEY", model_name=model)
                self.assertEqual(app.secrets.reads, [])

    def test_changed_resume_model_or_key_clears_stale_download(self):
        for label, value in ((TEXT_LABEL, "Different synthetic applicant"),
                             (MODEL_LABEL, list(MODEL_OPTIONS)[1]),
                             (KEY_LABEL, "DIFFERENT_SYNTHETIC_KEY")):
            with self.subTest(label=label):
                provider = self.provider()
                app = prepared_app(provider)
                app.click("Process resume")
                app.run(provider)
                app.change(label, value)
                app.run(provider)
                self.assertFalse(app.calls_of("download_button"))
                self.assertNotIn("resume_docx", app.session_state)
                provider.assert_called_once()

    def test_legacy_selection_shows_safe_error_without_export(self):
        provider = self.provider()
        provider.side_effect = SafeProcessingError("This model is retired. Select Luna or Terra.")
        app = prepared_app(provider)
        app.change(MODEL_LABEL, "fine-tuned-gpt-3.5-turbo-4k")
        app.click("Process resume")
        app.run(provider)
        self.assertIn("unavailable", app.calls_of("warning")[0]["value"])
        self.assertIn("retired", app.calls_of("error")[0]["value"])
        self.assertFalse(app.calls_of("download_button"))
        provider.return_value.extract_resume.assert_not_called()

    def test_failed_repeat_processing_does_not_keep_previous_download(self):
        provider = self.provider()
        provider.return_value.extract_resume.side_effect = [SECTIONS, RuntimeError("SENTINEL_PRIVATE_BODY")]
        app = prepared_app(provider)
        app.click("Process resume")
        app.run(provider)
        self.assertEqual(len(app.calls_of("download_button")), 2)
        app.click("Process resume")
        app.run(provider)
        self.assertFalse(app.calls_of("download_button"))
        self.assertNotIn("resume_docx", app.session_state)
        self.assertNotIn("SENTINEL", app.calls_of("error")[0]["value"])

    def test_safe_provider_error_stops_before_export(self):
        provider = self.provider()
        provider.return_value.extract_resume.side_effect = SafeProcessingError("OpenAI could not authenticate this key.")
        app = prepared_app(provider)
        app.click("Process resume")
        app.run(provider)
        self.assertIn("could not authenticate", app.calls_of("error")[0]["value"])
        self.assertFalse(app.calls_of("download_button"))

    def test_resume_output_is_plain_text_without_html_or_remote_markdown_images(self):
        provider = self.provider()
        payload = "<script>SYNTHETIC</script> ![sample](https://example.invalid/pixel)"
        provider.return_value.extract_resume.return_value = {**SECTIONS, "summary": payload}
        app = prepared_app(provider)
        app.click("Process resume")
        app.run(provider)
        rendered = [item for item in app.calls_of("markdown") if "<script>" in item["value"]]
        self.assertEqual(len(rendered), 0)
        self.assertIn(payload, [item["value"] for item in app.calls_of("text")])

    def test_clear_result_removes_both_downloads(self):
        provider = self.provider()
        app = prepared_app(provider)
        app.click("Process resume")
        app.run(provider)
        app.click("Clear result")
        app.run(provider)
        self.assertFalse(app.calls_of("download_button"))
        self.assertNotIn("resume_sections", app.session_state)
        provider.assert_called_once()


if __name__ == "__main__":
    unittest.main()
