"""Contract conformance, plus the parts of this engine that no contract can check.

Nothing here talks to a real endpoint: the transport is stubbed, so the suite runs in CI and on a
laptop. What a live model does with a real page is a question for a deployment, not for a test.
"""
import os
import unittest
from unittest.mock import patch

from escriptorium_engine_openai_api_compatible import OpenAIAPICompatibleEngine
from escriptorium_engine_openai_api_compatible.engine import (
    API_KEY_ENV,
    _clean,
    _headers,
    parse_reference,
)

from engines.base import EngineError, ModelValidationError
from engines.conformance import EngineConformanceMixin


class OpenAIAPICompatibleEngineConformanceTests(EngineConformanceMixin, unittest.TestCase):
    engine_class = OpenAIAPICompatibleEngine
    recognition_model = None      # inference needs an endpoint; those checks skip


class ReferenceParsingTests(unittest.TestCase):
    def test_url_and_model_are_split(self):
        base, model = parse_reference("http://localhost:11434/v1#qwen2.5vl:7b")
        self.assertEqual(base, "http://localhost:11434/v1")
        self.assertEqual(model, "qwen2.5vl:7b")

    def test_trailing_slash_is_normalised(self):
        base, _ = parse_reference("http://localhost:11434/v1/#m")
        self.assertEqual(base, "http://localhost:11434/v1")

    def test_a_model_name_is_required(self):
        """Without it there is nothing to ask, and the failure would surface only at transcription."""
        with self.assertRaisesRegex(ModelValidationError, "Name the model"):
            parse_reference("http://localhost:11434/v1")

    def test_only_http_is_accepted(self):
        with self.assertRaisesRegex(ModelValidationError, "not an http"):
            parse_reference("file:///etc/passwd#m")

    def test_empty_is_refused(self):
        with self.assertRaises(ModelValidationError):
            parse_reference("   ")


class AnswerCleaningTests(unittest.TestCase):
    """These are chat models: they wrap, quote and explain. Only wrappers may be removed."""

    def test_plain_text_is_untouched(self):
        self.assertEqual(_clean("Die Maschinen-Industrie"), "Die Maschinen-Industrie")

    def test_code_fences_are_stripped(self):
        self.assertEqual(_clean("```\nDie Maschinen\n```"), "Die Maschinen")

    def test_quotes_are_stripped(self):
        self.assertEqual(_clean('"Die Maschinen"'), "Die Maschinen")

    def test_a_preamble_is_stripped(self):
        self.assertEqual(_clean("The text reads: Die Maschinen"), "Die Maschinen")

    def test_inner_quotes_are_kept(self):
        """Only a wrapping pair goes; quotation inside the line is part of the transcription."""
        self.assertEqual(_clean('er sagte "halt" laut'), 'er sagte "halt" laut')

    def test_whitespace_is_collapsed_not_content(self):
        self.assertEqual(_clean("  zwei   Wörter \n"), "zwei Wörter")


class ValidationTests(unittest.TestCase):
    def test_an_uploaded_file_is_refused(self):
        with self.assertRaisesRegex(ModelValidationError, "not an uploaded file"):
            OpenAIAPICompatibleEngine().validate_model("/tmp/whatever.bin")

    def test_a_model_the_endpoint_does_not_serve_is_refused(self):
        with patch.object(OpenAIAPICompatibleEngine, "_list_models", return_value={"llama3.1:8b"}):
            with self.assertRaisesRegex(ModelValidationError, "does not serve"):
                OpenAIAPICompatibleEngine().validate_reference("http://host/v1#qwen2.5vl:7b")

    def test_a_served_model_is_accepted_and_normalised(self):
        with patch.object(OpenAIAPICompatibleEngine, "_list_models", return_value={"qwen2.5vl:7b"}):
            info = OpenAIAPICompatibleEngine().validate_reference("http://host/v1/#qwen2.5vl:7b")
        self.assertEqual(info.job, "recognition")
        self.assertEqual(info.reference, "http://host/v1#qwen2.5vl:7b")

    def test_an_endpoint_without_model_listing_is_still_accepted(self):
        """Not every gateway implements /models; only an unreachable endpoint is an error."""
        with patch.object(OpenAIAPICompatibleEngine, "_list_models", return_value=set()):
            info = OpenAIAPICompatibleEngine().validate_reference("http://host/v1#anything")
        self.assertEqual(info.reference, "http://host/v1#anything")


class CredentialTests(unittest.TestCase):
    """A local server needs no key; a hosted one does. Neither may leak into the model reference."""

    def test_no_key_configured_sends_no_authorization(self):
        """The case this was built against: a local endpoint that would reject an empty bearer."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("Authorization", _headers())

    def test_a_configured_key_is_sent_as_a_bearer_token(self):
        with patch.dict(os.environ, {API_KEY_ENV: "sk-example"}, clear=True):
            self.assertEqual(_headers()["Authorization"], "Bearer sk-example")

    def test_a_blank_key_is_treated_as_absent(self):
        """An unset variable in a compose file arrives as an empty string, not as missing."""
        with patch.dict(os.environ, {API_KEY_ENV: "   "}, clear=True):
            self.assertNotIn("Authorization", _headers())

    def test_the_content_type_is_always_present(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_headers()["Content-Type"], "application/json")

    def test_the_reference_is_not_a_place_for_a_key(self):
        """The reference is stored in the database and shown in the interface, so a secret put
        there would be exposed. Parsing keeps only the endpoint and the model name."""
        base, model = parse_reference("http://host/v1#qwen2.5vl:7b")
        self.assertEqual(base, "http://host/v1")
        self.assertEqual(model, "qwen2.5vl:7b")


class TransportFailureTests(unittest.TestCase):
    """A broken endpoint must fail the request, not quietly produce empty transcriptions."""

    def test_a_bad_status_raises_engine_error(self):
        class Response:
            ok = False
            status_code = 503
            text = "upstream is down"

        with patch("requests.post", return_value=Response()):
            with self.assertRaisesRegex(EngineError, "503"):
                OpenAIAPICompatibleEngine._ask("http://host/v1", "m", "data:image/png;base64,AA==")

    def test_an_unexpected_body_raises_engine_error(self):
        class Response:
            ok = True
            status_code = 200

            def json(self):
                return {"no": "choices"}

        with patch("requests.post", return_value=Response()):
            with self.assertRaisesRegex(EngineError, "unexpected answer"):
                OpenAIAPICompatibleEngine._ask("http://host/v1", "m", "data:image/png;base64,AA==")

    def test_a_normal_answer_is_returned_cleaned(self):
        class Response:
            ok = True
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": '"Die Maschinen"'}}]}

        with patch("requests.post", return_value=Response()):
            text = OpenAIAPICompatibleEngine._ask("http://host/v1", "m", "data:image/png;base64,AA==")
        self.assertEqual(text, "Die Maschinen")


class VersionSourceTests(unittest.TestCase):
    def test_the_model_name_is_recorded_not_the_host(self):
        """The host is deployment detail and changes; the model is what produced the text."""
        class Model:
            reference = "http://localhost:11434/v1#qwen2.5vl:7b"

        self.assertEqual(OpenAIAPICompatibleEngine().version_source(Model()), "openai-api-compatible:qwen2.5vl:7b")
