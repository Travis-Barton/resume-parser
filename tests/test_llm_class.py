"""Synthetic-only tests for the original retrieved-example extraction method."""
import contextlib
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import httpx2
from openai import OpenAI
from llm_class import (INDEX_PATH, LLMUtils, MAX_RESUME_CHARACTERS, MODEL_OPTIONS,
                       SECTIONS, SUPPORTED_MODELS, ResumeProcessingError)


def example(label, **overrides):
    metadata = {"resume": f"SYNTHETIC_SOURCE_{label}"}
    metadata.update({s: f"SYNTHETIC_{s.upper()}_{label}" for s in SECTIONS if s != "name"})
    metadata.update(overrides)
    return SimpleNamespace(metadata=metadata)


def source(*documents):
    return SimpleNamespace(search=Mock(return_value=list(documents) or [example("ONE"), example("TWO")]))


def fake_client(content="SYNTHETIC_SECTION", *, finish_reason="stop", refusal=None):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal))], _request_id="req_synthetic")
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))


def synthetic_chat_response(model="gpt-6-luna"):
    return httpx2.Response(200, json={"id": "chatcmpl_synthetic", "object": "chat.completion",
        "created": 1, "model": model, "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": "SYNTHETIC_SECTION"}}]})


class ExtractionTests(unittest.TestCase):
    def test_picker_extends_legacy_choices_with_supported_models(self):
        self.assertEqual(set(SUPPORTED_MODELS), {"gpt-6-luna", "gpt-6.1-terra"})
        self.assertTrue(set(SUPPORTED_MODELS).issubset(MODEL_OPTIONS.values()))
        for legacy in ("fine-tuned-gpt-3.5-turbo-4k", "fine_tuned_gpt-3.5-turbo-16k-no-dot", "gpt-3.5-turbo-16k"):
            self.assertIn(legacy, MODEL_OPTIONS)

    def test_each_model_extracts_five_raw_sections_with_one_retrieval(self):
        for model in SUPPORTED_MODELS:
            with self.subTest(model=model):
                client, examples = fake_client(), source()
                result = LLMUtils("SYNTHETIC_KEY", model, client=client, embedding_source=examples).extract_resume("SYNTHETIC_APPLICANT")
                self.assertEqual(result, dict.fromkeys(SECTIONS, "SYNTHETIC_SECTION"))
                examples.search.assert_called_once_with("SYNTHETIC_APPLICANT", k=100, search_type="similarity")
                calls = client.chat.completions.create.call_args_list
                self.assertEqual(len(calls), 5)
                for index, call in enumerate(calls):
                    args = call.kwargs
                    self.assertEqual(args["model"], model)
                    self.assertEqual(args["reasoning_effort"], "none")
                    self.assertEqual(args["max_completion_tokens"], 128 if index == 0 else 8192)
                    self.assertIs(args["store"], False)
                    self.assertEqual([m["role"] for m in args["messages"]], ["system", "user"])
                    for field in ("response_format", "temperature", "max_tokens"):
                        self.assertNotIn(field, args)
                    self.assertNotIn("SYNTHETIC_KEY", json.dumps(args))

    def test_original_prompt_filters_then_selects_top_two_metadata_examples(self):
        examples = source(example("EMPTY", summary=""), example("NULL", summary=None),
            example("NONE", summary="None"), example("ONE"), example("TWO"), example("THREE"))
        client = fake_client()
        LLMUtils("SYNTHETIC", client=client, embedding_source=examples).extract_section("SYNTHETIC_APPLICANT", "summary")
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        prompt = messages[0]["content"]
        self.assertIn("\nExample 1:\n\n```resume\nSYNTHETIC_SOURCE_ONE\n```\n```summary\nSYNTHETIC_SUMMARY_ONE\n```\n", prompt)
        self.assertIn("Example 2:", prompt)
        self.assertIn("SYNTHETIC_SUMMARY_TWO", prompt)
        self.assertNotIn("Example 3:", prompt)
        for marker in ("SOURCE_EMPTY", "SOURCE_NULL", "SOURCE_NONE", "SOURCE_THREE"):
            self.assertNotIn(marker, prompt)
        self.assertIn("Never include items that are not in the applicant's resume", prompt)
        self.assertIn("These are examples, never include them in the extracted section.", prompt)
        self.assertIn("```resume\nSYNTHETIC_APPLICANT\n```", messages[1]["content"])

    def test_section_prompts_keep_historical_experience_and_skill_rules(self):
        client = fake_client()
        util = LLMUtils("SYNTHETIC", client=client, embedding_source=source())
        util.extract_section("SYNTHETIC", "professional_experience")
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("PROFESSIONAL EXPERIENCE does NOT include education", prompt)
        self.assertIn("153 characters long exactly", prompt)
        util.extract_section("SYNTHETIC", "skills_and_tech")
        self.assertIn("Skills and Tech only includes bullet points of skills and tech",
                      client.chat.completions.create.call_args.kwargs["messages"][0]["content"])

    def test_retrieval_cache_preserves_metadata_and_changes_with_applicant(self):
        client, examples = fake_client(), source()
        util = LLMUtils("SYNTHETIC", client=client, embedding_source=examples)
        util.extract_section("SYNTHETIC_FIRST", "summary")
        util.extract_section("SYNTHETIC_FIRST", "education")
        self.assertEqual(examples.search.call_count, 1)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("SYNTHETIC_EDUCATION_ONE", prompt)
        self.assertNotIn("SYNTHETIC_SUMMARY_ONE", prompt)
        util.extract_section("SYNTHETIC_SECOND", "education")
        self.assertEqual(examples.search.call_count, 2)

    def test_name_uses_selected_model_without_retrieval_or_examples(self):
        client, examples = fake_client("Synthetic Name"), source()
        util = LLMUtils("SYNTHETIC", "gpt-6.1-terra", client=client, embedding_source=examples)
        self.assertEqual(util.extract_section("SYNTHETIC_APPLICANT", "name"), "Synthetic Name")
        examples.search.assert_not_called()
        args = client.chat.completions.create.call_args.kwargs
        self.assertEqual(args["model"], "gpt-6.1-terra")
        self.assertEqual(args["max_completion_tokens"], 128)
        self.assertNotIn("SYNTHETIC_SOURCE", json.dumps(args))

    def test_missing_key_and_legacy_model_fail_before_client_or_retrieval(self):
        examples = source()
        with patch("llm_class.OpenAI") as factory:
            cases = [("", "gpt-6-luna"), (None, "gpt-6-luna"), ("SYNTHETIC", "unsupported")]
            cases.extend(("SYNTHETIC", m) for m in MODEL_OPTIONS.values() if m not in SUPPORTED_MODELS)
            for key, model in cases:
                with self.assertRaises(ResumeProcessingError) as caught:
                    LLMUtils(key, model, embedding_source=examples)
                self.assertEqual(caught.exception.category, "configuration")
            factory.assert_not_called()
        examples.search.assert_not_called()

    def test_empty_oversize_and_invalid_section_make_no_request(self):
        client, examples = fake_client(), source()
        util = LLMUtils("SYNTHETIC", client=client, embedding_source=examples)
        for value in ("", " ", None, "x" * (MAX_RESUME_CHARACTERS + 1)):
            with self.assertRaises(ResumeProcessingError) as caught:
                util.extract_resume(value)
            self.assertEqual(caught.exception.category, "input")
        with self.assertRaises(ResumeProcessingError):
            util.extract_section("SYNTHETIC", "unsupported")
        client.chat.completions.create.assert_not_called()
        examples.search.assert_not_called()

    def test_historical_index_load_is_lazy_and_path_is_not_user_controlled(self):
        examples, client = source(), fake_client()
        with patch("langchain.vectorstores.FAISS.load_local", return_value=examples) as loader:
            util = LLMUtils("SYNTHETIC", client=client)
            loader.assert_not_called()
            util.extract_section("SYNTHETIC", "summary")
            self.assertEqual(loader.call_args.args, (str(INDEX_PATH),))
            self.assertIs(loader.call_args.kwargs["embeddings"].client, client)
            util.extract_section("SYNTHETIC", "education")
            loader.assert_called_once()

    def test_index_load_errors_are_sanitized_without_chat_call(self):
        client = fake_client()
        with patch("langchain.vectorstores.FAISS.load_local", side_effect=RuntimeError("SENSITIVE_SENTINEL")):
            with self.assertRaises(ResumeProcessingError) as caught:
                LLMUtils("SYNTHETIC", client=client).extract_resume("SYNTHETIC")
        self.assertEqual(caught.exception.category, "examples")
        self.assertNotIn("SENSITIVE_SENTINEL", str(caught.exception))
        client.chat.completions.create.assert_not_called()

    def test_refusal_truncation_and_invalid_xml_text_never_export(self):
        cases = [(fake_client(refusal="SENSITIVE_SENTINEL"), "refusal"),
            (fake_client(finish_reason="length"), "incomplete"),
            (fake_client(finish_reason="content_filter"), "incomplete"),
            (fake_client(content=None), "invalid_output"),
            (fake_client(content="\ud800"), "invalid_output"),
            (fake_client(content="\ufffe"), "invalid_output"),
            (fake_client(content="x" * (MAX_RESUME_CHARACTERS + 1)), "invalid_output")]
        for client, category in cases:
            with self.assertRaises(ResumeProcessingError) as caught:
                LLMUtils("SYNTHETIC", client=client, embedding_source=source()).extract_resume("SYNTHETIC")
            self.assertEqual(caught.exception.category, category)
            self.assertNotIn("SENSITIVE_SENTINEL", str(caught.exception))
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_blank_sections_and_valid_unicode_are_returned_as_text(self):
        for content in ("", "Name Not Found", "Synthetic \U0001f680\nParagraph"):
            result = LLMUtils("SYNTHETIC", client=fake_client(content), embedding_source=source()).extract_section("SYNTHETIC", "summary")
            self.assertEqual(result, content)

    def test_provider_and_retrieval_errors_are_sanitized(self):
        for stage in ("chat", "retrieval"):
            for status, code, category in [(401, "invalid_api_key", "authentication"),
                (403, None, "model_access"), (404, "model_not_found", "model_access"),
                (400, "model_deprecated", "model_access"), (429, "insufficient_quota", "quota"),
                (429, "rate_limit_exceeded", "rate_limit"), (503, None, "provider")]:
                with self.subTest(stage=stage, category=category):
                    error = RuntimeError("KEY_SENTINEL RESUME_SENTINEL")
                    error.status_code, error.body, error.request_id = status, {"error": {"code": code}}, "req_synthetic"
                    client, examples = fake_client(), source()
                    target = client.chat.completions.create if stage == "chat" else examples.search
                    target.side_effect = error
                    capture = io.StringIO()
                    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                        with self.assertRaises(ResumeProcessingError) as caught:
                            LLMUtils("KEY_SENTINEL", client=client, embedding_source=examples).extract_resume("RESUME_SENTINEL")
                    self.assertEqual(caught.exception.category, category)
                    for marker in ("KEY_SENTINEL", "RESUME_SENTINEL"):
                        self.assertNotIn(marker, str(caught.exception) + capture.getvalue())
                    self.assertTrue(caught.exception.__suppress_context__)

    def test_transport_errors_and_untrusted_request_reference_are_sanitized(self):
        for name, category in [("APITimeoutError", "timeout"), ("APIConnectionError", "connection")]:
            error = type(name, (Exception,), {})("SENSITIVE_SENTINEL")
            error.request_id = "SENSITIVE_SENTINEL\nunsafe"
            client = fake_client()
            client.chat.completions.create.side_effect = error
            with self.assertRaises(ResumeProcessingError) as caught:
                LLMUtils("SYNTHETIC", client=client, embedding_source=source()).extract_section("SYNTHETIC", "name")
            self.assertEqual(caught.exception.category, category)
            self.assertNotIn("SENSITIVE_SENTINEL", str(caught.exception))

    def test_real_sdk_serializes_original_chat_method_without_schema(self):
        requests = []
        def handle(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            payload = json.loads(request.content)
            requests.append(payload)
            return synthetic_chat_response(payload["model"])
        with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
            with OpenAI(api_key="SYNTHETIC", http_client=transport, max_retries=0) as sdk:
                LLMUtils("SYNTHETIC", client=sdk, embedding_source=source()).extract_resume("SYNTHETIC")
        self.assertEqual(len(requests), 5)
        for payload in requests:
            self.assertNotIn("response_format", payload)
            self.assertEqual(payload["reasoning_effort"], "none")
            self.assertIs(payload["store"], False)

    def test_app_created_sdk_does_not_retry_provider_503(self):
        attempts = []
        def handle(request):
            attempts.append(request.url.path)
            return httpx2.Response(503, json={"error": {"code": "synthetic_failure", "message": "SENSITIVE_SENTINEL"}})
        with patch.dict(os.environ, {}, clear=True), contextlib.ExitStack() as stack:
            transport = stack.enter_context(httpx2.Client(transport=httpx2.MockTransport(handle)))
            def factory(**kwargs):
                return stack.enter_context(OpenAI(http_client=transport, **kwargs))
            with patch("llm_class.OpenAI", side_effect=factory):
                with self.assertRaises(ResumeProcessingError) as caught:
                    LLMUtils("SYNTHETIC", embedding_source=source()).extract_section("SYNTHETIC", "name")
        self.assertEqual(attempts, ["/v1/chat/completions"])
        self.assertEqual(caught.exception.category, "provider")
        self.assertNotIn("SENSITIVE_SENTINEL", str(caught.exception))

    def test_separate_clients_keep_entered_keys_and_official_endpoint(self):
        observed = {"first": [], "second": []}
        def handler(label):
            def handle(request):
                observed[label].append((request.headers["Authorization"], request.url.host))
                return synthetic_chat_response(json.loads(request.content)["model"])
            return handle
        with patch.dict(os.environ, {"OPENAI_API_KEY": "SYNTHETIC_ENV", "OPENAI_BASE_URL": "https://untrusted.invalid/v1"}, clear=True), contextlib.ExitStack() as stack:
            transports = iter([stack.enter_context(httpx2.Client(transport=httpx2.MockTransport(handler(label)))) for label in ("first", "second")])
            def factory(**kwargs):
                self.assertEqual(kwargs["max_retries"], 0)
                return stack.enter_context(OpenAI(http_client=next(transports), **kwargs))
            with patch("llm_class.OpenAI", side_effect=factory):
                first = LLMUtils("SYNTHETIC_FIRST", embedding_source=source())
                second = LLMUtils("SYNTHETIC_SECOND", "gpt-6.1-terra", embedding_source=source())
                first.extract_section("SYNTHETIC", "name")
                second.extract_section("SYNTHETIC", "name")
                first.extract_section("SYNTHETIC", "name")
        self.assertEqual(observed["first"], [("Bearer SYNTHETIC_FIRST", "api.openai.com")] * 2)
        self.assertEqual(observed["second"], [("Bearer SYNTHETIC_SECOND", "api.openai.com")])


if __name__ == "__main__":
    unittest.main()
