"""Resume extraction with explicit per-client credentials and structured output."""
import json
import re

from openai import OpenAI

from prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from resume_contract import RESUME_JSON_SCHEMA, to_section_results, validate_resume

MODEL_OPTIONS = {
    "Luna — few-shot extraction": "gpt-6-luna",
    "Terra — few-shot extraction": "gpt-6.1-terra",
}
MAX_RESUME_CHARACTERS = 100_000
MAX_OUTPUT_TOKENS = 8192


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


class LLMUtils:
    def __init__(self, api_key, model_name="gpt-6-luna", *, client=None):
        if model_name not in MODEL_OPTIONS.values():
            raise ResumeProcessingError("configuration", "Select one of the available extraction models.")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ResumeProcessingError("configuration", "Enter your OpenAI API key before processing.")
        self.model_name = model_name
        self.client = client if client is not None else OpenAI(
            api_key=api_key.strip(), timeout=60.0, max_retries=0,
        )

    def extract_resume(self, resume):
        if not isinstance(resume, str) or not resume.strip():
            raise ResumeProcessingError("input", "The resume is empty. Please provide readable text.")
        if len(resume) > MAX_RESUME_CHARACTERS:
            raise ResumeProcessingError("input", "The resume is too large. Limit the input to 100,000 characters.")
        messages = [{"role": "developer", "content": SYSTEM_PROMPT}]
        for example in FEW_SHOT_EXAMPLES:
            messages.extend([
                {"role": "user", "content": example["resume"]},
                {"role": "assistant", "content": json.dumps(example["result"], ensure_ascii=False)},
            ])
        messages.append({"role": "user", "content": resume})
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
                reasoning={"effort": "none"},
                text={"format": {
                    "type": "json_schema", "name": "aim_resume",
                    "schema": RESUME_JSON_SCHEMA, "strict": True,
                }},
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
            )
        except Exception as exc:
            raise _safe_provider_error(exc) from None

        request_id = _field(response, "_request_id")
        for output in _field(response, "output", []) or []:
            if _field(output, "type") == "message":
                for content in _field(output, "content", []) or []:
                    if _field(content, "type") == "refusal":
                        raise ResumeProcessingError(
                            "refusal", "The model declined this extraction. No document was generated.", request_id,
                        )
        if _field(response, "status") != "completed":
            raise ResumeProcessingError(
                "incomplete", "The extraction did not finish. Try a shorter resume; no partial document was generated.", request_id,
            )
        try:
            data = json.loads(_field(response, "output_text", ""))
            validate_resume(data)
            return to_section_results(data)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ResumeProcessingError(
                "invalid_output", "The model returned an invalid resume structure. No document was generated.", request_id,
            ) from None
