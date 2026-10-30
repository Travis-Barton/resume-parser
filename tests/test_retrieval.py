"""Offline embedding/index tests. Never load the repository's historical pickle."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import faiss
import httpx2
import numpy as np
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from openai import OpenAI

from llm_class import LegacyIndexEmbeddings, LLMUtils, ResumeProcessingError


def vector(first=1.0, second=0.0):
    return [first, second] + [0.0] * 1534


def row(index, values):
    return SimpleNamespace(index=index, embedding=values)


def tokenizer(tokens):
    return patch("llm_class.tiktoken.get_encoding", return_value=SimpleNamespace(encode=Mock(return_value=tokens)))


def embedding_client(rows):
    return SimpleNamespace(embeddings=SimpleNamespace(create=Mock(return_value=SimpleNamespace(data=rows))))


def metadata(label):
    return {"resume": f"SYNTHETIC_SOURCE_{label}", "summary": f"SYNTHETIC_SUMMARY_{label}",
        "skills_and_tech": "Synthetic skill", "professional_experience": "Synthetic role",
        "education": "Synthetic education"}


def chat_response(model):
    return httpx2.Response(200, json={"id": "chatcmpl_synthetic", "object": "chat.completion",
        "created": 1, "model": model, "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": "SYNTHETIC_SECTION"}}]})


class EmbeddingTests(unittest.TestCase):
    def test_single_chunk_uses_original_model_tokens_and_normalizes(self):
        client = embedding_client([row(0, vector(3, 4))])
        with tokenizer([10, 20, 30]) as get_encoding:
            result = LegacyIndexEmbeddings(client).embed_query("SYNTHETIC_QUERY")
        get_encoding.assert_called_once_with("cl100k_base")
        get_encoding.return_value.encode.assert_called_once_with("SYNTHETIC_QUERY", disallowed_special=())
        client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002", input=[[10, 20, 30]], encoding_format="float")
        self.assertEqual(len(result), 1536)
        self.assertAlmostEqual(result[0], 0.6)
        self.assertAlmostEqual(result[1], 0.8)
        self.assertAlmostEqual(sum(x * x for x in result), 1.0)

    def test_multiple_chunks_are_one_batch_and_weighted_by_token_count(self):
        tokens = list(range(8193))
        client = embedding_client([row(1, vector(0, 1)), row(0, vector(1, 0))])
        with tokenizer(tokens):
            result = LegacyIndexEmbeddings(client).embed_query("SYNTHETIC_LONG_QUERY")
        client.embeddings.create.assert_called_once()
        self.assertEqual(client.embeddings.create.call_args.kwargs["input"], [tokens[:8191], tokens[8191:]])
        denominator = math.sqrt(8191 ** 2 + 2 ** 2)
        self.assertAlmostEqual(result[0], 8191 / denominator)
        self.assertAlmostEqual(result[1], 2 / denominator)

    def test_exact_chunk_boundary_and_empty_input(self):
        client = embedding_client([row(0, vector())])
        with tokenizer([1] * 8191):
            LegacyIndexEmbeddings(client).embed_query("SYNTHETIC_BOUNDARY")
        self.assertEqual(len(client.embeddings.create.call_args.kwargs["input"]), 1)
        empty = embedding_client([])
        with tokenizer([]), self.assertRaises(ResumeProcessingError) as caught:
            LegacyIndexEmbeddings(empty).embed_query("")
        self.assertEqual(caught.exception.category, "input")
        empty.embeddings.create.assert_not_called()

    def test_invalid_embedding_rows_are_rejected(self):
        cases = [[], [row(1, vector())], [row(0, vector()), row(0, vector())],
                 [row(0, [1.0])], [row(0, vector(float("nan")))], [row(0, vector(0, 0))]]
        for rows in cases:
            with self.subTest(row_count=len(rows)), tokenizer([1, 2]):
                with self.assertRaises(ResumeProcessingError) as caught:
                    LegacyIndexEmbeddings(embedding_client(rows)).embed_query("SYNTHETIC_QUERY")
                self.assertEqual(caught.exception.category, "examples")

    def test_synthetic_faiss_roundtrip_retrieves_without_historical_index(self):
        client = embedding_client([row(0, vector())])
        completion = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content="SYNTHETIC_SECTION", refusal=None))])
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=completion)))
        adapter = LegacyIndexEmbeddings(client)
        index = faiss.IndexFlatL2(1536)
        index.add(np.asarray([vector(), vector(0, 1)], dtype=np.float32))
        documents = InMemoryDocstore({label: Document(page_content="SYNTHETIC", metadata=metadata(label)) for label in ("ONE", "TWO")})
        database = FAISS(adapter.embed_query, index, documents, {0: "ONE", 1: "TWO"})
        # Use the writable checkout: Windows sandbox TEMP can point outside its
        # writable roots, which the native FAISS writer cannot access.
        temporary_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="resume-synthetic-faiss-", dir=temporary_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(temporary_root))
            database.save_local(directory)
            loaded = FAISS.load_local(directory, embeddings=adapter)
            with tokenizer([10, 20]):
                result = LLMUtils("SYNTHETIC", client=client, embedding_source=loaded).extract_resume("SYNTHETIC_APPLICANT")
        client.embeddings.create.assert_called_once()
        self.assertEqual(client.chat.completions.create.call_count, 5)
        prompt = client.chat.completions.create.call_args_list[1].kwargs["messages"][0]["content"]
        self.assertLess(prompt.index("SYNTHETIC_SOURCE_ONE"), prompt.index("SYNTHETIC_SOURCE_TWO"))
        self.assertEqual(result["summary"], "SYNTHETIC_SECTION")

    def test_real_sdk_embedding_and_chat_share_only_their_own_entered_key(self):
        observed = {"first": [], "second": []}
        def handler(label):
            def handle(request):
                payload = json.loads(request.content)
                observed[label].append((request.url.path, request.headers["Authorization"], request.url.host))
                if request.url.path == "/v1/embeddings":
                    self.assertEqual(payload["model"], "text-embedding-ada-002")
                    self.assertEqual(payload["encoding_format"], "float")
                    return httpx2.Response(200, json={"object": "list", "model": "text-embedding-ada-002",
                        "data": [{"object": "embedding", "index": 0, "embedding": vector()}],
                        "usage": {"prompt_tokens": 2, "total_tokens": 2}})
                self.assertEqual(request.url.path, "/v1/chat/completions")
                return chat_response(payload["model"])
            return handle
        with patch.dict(os.environ, {"OPENAI_API_KEY": "SYNTHETIC_ENV"}, clear=True), contextlib.ExitStack() as stack:
            transports = iter([stack.enter_context(httpx2.Client(transport=httpx2.MockTransport(handler(label)))) for label in ("first", "second")])
            def factory(**kwargs):
                return stack.enter_context(OpenAI(http_client=next(transports), **kwargs))
            with patch("llm_class.OpenAI", side_effect=factory), tokenizer([10, 20]):
                first = LLMUtils("SYNTHETIC_FIRST")
                second = LLMUtils("SYNTHETIC_SECOND", "gpt-6.1-terra")
                for util in (first, second):
                    LegacyIndexEmbeddings(util.client).embed_query("SYNTHETIC_QUERY")
                    util.extract_section("SYNTHETIC_APPLICANT", "name")
        for label, credential in (("first", "SYNTHETIC_FIRST"), ("second", "SYNTHETIC_SECOND")):
            self.assertEqual(observed[label], [(path, "Bearer " + credential, "api.openai.com")
                for path in ("/v1/embeddings", "/v1/chat/completions")])

    def test_real_sdk_embedding_503_is_not_retried_and_stops_chat(self):
        attempts = []
        def handle(request):
            attempts.append(request.url.path)
            return httpx2.Response(503, json={"error": {"code": "synthetic_failure", "message": "SENSITIVE_SENTINEL"}})
        capture = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.ExitStack() as stack:
            transport = stack.enter_context(httpx2.Client(transport=httpx2.MockTransport(handle)))
            def factory(**kwargs):
                return stack.enter_context(OpenAI(http_client=transport, **kwargs))
            with patch("llm_class.OpenAI", side_effect=factory), tokenizer([10, 20]):
                util = LLMUtils("SYNTHETIC")
                def search(resume, **kwargs):
                    LegacyIndexEmbeddings(util.client).embed_query(resume)
                    return []
                util.embedding_source = SimpleNamespace(search=search)
                with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                    with self.assertRaises(ResumeProcessingError) as caught:
                        util.extract_resume("SYNTHETIC_APPLICANT")
        self.assertEqual(attempts, ["/v1/embeddings"])
        self.assertEqual(caught.exception.category, "provider")
        self.assertNotIn("SENSITIVE_SENTINEL", str(caught.exception) + capture.getvalue())


if __name__ == "__main__":
    unittest.main()
