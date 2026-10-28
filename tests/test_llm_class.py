"""Offline request, credential-isolation, and sanitized-error regressions."""
import copy
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import httpx2
from openai import OpenAI

from llm_class import LLMUtils, MODEL_OPTIONS, ResumeProcessingError, MAX_RESUME_CHARACTERS
from prompts import FEW_SHOT_EXAMPLES
from resume_contract import to_section_results, validate_resume


class ExtractionTests(unittest.TestCase):
    def test_real_sdk_serialization_with_in_memory_transport(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json={
                "id": "resp_synthetic", "object": "response", "created_at": 1,
                "model": "gpt-6-luna", "status": "completed",
                "output": [{"id": "msg_synthetic", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text", "annotations": [],
                        "text": json.dumps(FEW_SHOT_EXAMPLES[0]["result"])}]}],
            })

        with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
            with OpenAI(api_key="SYNTHETIC_ONLY", http_client=transport, max_retries=0) as sdk:
                sections = LLMUtils("SYNTHETIC_ONLY", client=sdk).extract_resume("SYNTHETIC_RESUME")
        self.assertEqual(sections["name"], "Jordan Example")
        self.assertEqual(len(requests), 1)
        self.assertFalse(requests[0]["store"])
        self.assertEqual(requests[0]["text"]["format"]["type"], "json_schema")

    def client(self, *, status="completed", output=None, data=None, text=None):
        result = SimpleNamespace(
            status=status, output=[] if output is None else output,
            output_text=text if text is not None else json.dumps(
                FEW_SHOT_EXAMPLES[0]["result"] if data is None else data
            ),
            _request_id="req_synthetic",
        )
        return SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=result)))

    def test_each_supported_model_uses_strict_nonstored_response_and_all_examples(self):
        for model in MODEL_OPTIONS.values():
            with self.subTest(model=model):
                client = self.client()
                result = LLMUtils("SYNTHETIC_KEY", model, client=client).extract_resume("SYNTHETIC_APPLICANT")
                self.assertEqual(result, to_section_results(FEW_SHOT_EXAMPLES[0]["result"]))
                call = client.responses.create.call_args.kwargs
                self.assertEqual(call["model"], model)
                self.assertEqual(call["reasoning"], {"effort": "none"})
                self.assertFalse(call["store"])
                self.assertTrue(call["text"]["format"]["strict"])
                self.assertEqual(call["input"][-1]["content"], "SYNTHETIC_APPLICANT")
                self.assertEqual(len(call["input"]), 2 + 2 * len(FEW_SHOT_EXAMPLES))
                self.assertNotIn("temperature", call)
                self.assertNotIn("SYNTHETIC_KEY", json.dumps(call))

    def test_own_client_per_credential_without_global_key_or_secret_lookup(self):
        with patch("llm_class.OpenAI") as factory:
            first = LLMUtils("SYNTHETIC_A", "gpt-6-luna")
            second = LLMUtils("SYNTHETIC_B", "gpt-6.1-terra")
            self.assertEqual(factory.call_args_list[0].kwargs["api_key"], "SYNTHETIC_A")
            self.assertEqual(factory.call_args_list[1].kwargs["api_key"], "SYNTHETIC_B")
            self.assertEqual(factory.call_count, 2)

    def test_missing_key_and_retired_model_fail_before_client_construction(self):
        with patch("llm_class.OpenAI") as factory:
            for key, model in [("", "gpt-6-luna"), ("SYNTHETIC", "ft:retired")]:
                with self.assertRaises(ResumeProcessingError):
                    LLMUtils(key, model)
            factory.assert_not_called()

    def test_empty_and_oversize_input_make_no_request(self):
        client = self.client()
        util = LLMUtils("SYNTHETIC", client=client)
        for value in ("", " ", None, "x" * (MAX_RESUME_CHARACTERS + 1)):
            with self.assertRaises(ResumeProcessingError) as caught:
                util.extract_resume(value)
            self.assertEqual(caught.exception.category, "input")
        client.responses.create.assert_not_called()

    def test_incomplete_output_is_not_accepted_even_when_json_is_valid(self):
        for status in ("incomplete", "failed", "in_progress", None):
            client = self.client(status=status)
            with self.assertRaises(ResumeProcessingError) as caught:
                LLMUtils("SYNTHETIC", client=client).extract_resume("SYNTHETIC")
            self.assertEqual(caught.exception.category, "incomplete")

    def test_refusal_is_detected_even_with_valid_output_text(self):
        client = self.client(output=[{"type": "message", "content": [
            {"type": "refusal", "refusal": "SENSITIVE_SENTINEL"}
        ]}])
        with self.assertRaises(ResumeProcessingError) as caught:
            LLMUtils("SYNTHETIC", client=client).extract_resume("SYNTHETIC")
        self.assertEqual(caught.exception.category, "refusal")
        self.assertNotIn("SENSITIVE_SENTINEL", str(caught.exception))

    def test_invalid_json_empty_output_and_wrong_schema_do_not_export(self):
        for text in ("", "{", "null", "[]", '{"name":"SYNTHETIC"}'):
            with self.subTest(text=text):
                with self.assertRaises(ResumeProcessingError) as caught:
                    LLMUtils("SYNTHETIC", client=self.client(text=text)).extract_resume("SYNTHETIC")
                self.assertEqual(caught.exception.category, "invalid_output")

    def test_error_categories_do_not_expose_provider_body_or_resume(self):
        cases = [
            (401, "invalid_api_key", "authentication"),
            (403, None, "model_access"), (404, "model_not_found", "model_access"),
            (400, "model_deprecated", "model_access"),
            (429, "insufficient_quota", "quota"), (429, "rate_limit_exceeded", "rate_limit"),
            (500, None, "provider"),
        ]
        for status, code, category in cases:
            with self.subTest(category=category):
                error = RuntimeError("KEY_SENTINEL RESUME_SENTINEL")
                error.status_code = status
                error.body = {"error": {"code": code, "message": "KEY_SENTINEL"}}
                error.request_id = "req_synthetic"
                client = self.client()
                client.responses.create.side_effect = error
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                    with self.assertRaises(ResumeProcessingError) as caught:
                        LLMUtils("KEY_SENTINEL", client=client).extract_resume("RESUME_SENTINEL")
                self.assertEqual(caught.exception.category, category)
                self.assertIn("req_synthetic", str(caught.exception))
                self.assertNotIn("KEY_SENTINEL", str(caught.exception) + capture.getvalue())
                self.assertNotIn("RESUME_SENTINEL", str(caught.exception) + capture.getvalue())
                self.assertTrue(caught.exception.__suppress_context__)

    def test_transport_errors_and_untrusted_request_id_are_sanitized(self):
        for name, category in [("APITimeoutError", "timeout"), ("APIConnectionError", "connection")]:
            error = type(name, (Exception,), {})("SENSITIVE")
            error.request_id = "SENSITIVE\nunsafe"
            client = self.client()
            client.responses.create.side_effect = error
            with self.assertRaises(ResumeProcessingError) as caught:
                LLMUtils("SYNTHETIC", client=client).extract_resume("SYNTHETIC")
            self.assertEqual(caught.exception.category, category)
            self.assertNotIn("SENSITIVE", str(caught.exception))

    def test_all_synthetic_prompt_examples_validate_and_convert(self):
        for example in FEW_SHOT_EXAMPLES:
            validate_resume(copy.deepcopy(example["result"]))
            self.assertEqual(set(to_section_results(example["result"])), {
                "name", "summary", "skills_and_tech", "professional_experience", "education"
            })


if __name__ == "__main__":
    unittest.main()
