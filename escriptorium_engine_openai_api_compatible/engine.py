"""A vision-language model behind an OpenAI-compatible API, as an eScriptorium engine.

The model is not a file and not a repository: it is an endpoint plus the name of a model served
there. That is what `SOURCE_ENDPOINT` is for, and this engine is its first user. The same engine
therefore talks to a local Ollama or vLLM instance and to a hosted service; only the URL differs.

The engine is named after the interface it speaks, not after the kind of model behind it: anything
reachable at /v1/chat/completions that accepts an image works. In practice that is a vision-language
model, which reads a line as language rather than as shapes and so copes with scripts and layouts no
one trained a model for. What it does not give you: character positions, confidences, or any
guarantee that the answer is a transcription at all.
A model that decides to explain the image instead of reading it is a failure mode that does not
exist in kraken, and `_clean` exists to catch its most common forms.

One request per line, deliberately. Batching would be faster, but a model answering about several
images at once routinely mixes them up, and a silently mismatched line is worse than a slow one.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from collections.abc import Iterable, Iterator
from urllib.parse import urlsplit, urlunsplit

from engines.base import (
    SOURCE_ENDPOINT,
    BaseEngine,
    EngineError,
    EngineSpec,
    LineInput,
    ModelInfo,
    ModelValidationError,
    RecognitionResult,
    RecognizeOptions,
)
from engines.imaging import crop_line

logger = logging.getLogger(__name__)

#: what the model is asked to do. Kept blunt: every added instruction is another thing the model can
#: decide to talk about instead of transcribing.
PROMPT = (
    "Transcribe the text in this image exactly as written. "
    "Reply with the transcription only: no explanation, no quotation marks, no markdown. "
    "If the image contains no legible text, reply with an empty line."
)

#: Endpoints that need no credentials -- a local Ollama or vLLM -- are the common case, so a key is
#: optional. It is read from the environment rather than from the model reference on purpose: the
#: reference is stored in the database and shown in the interface, which is no place for a secret.
API_KEY_ENV = "ESCRIPTORIUM_OPENAI_API_KEY"

REQUEST_TIMEOUT = 120      # seconds per line; a cold model on the other side can be slow
MAX_EDGE = 1600            # px; longest edge of a line image sent, to bound payload size
MAX_TOKENS = 512


def parse_reference(reference: str) -> tuple[str, str]:
    """Split `http://host:port/v1#model` into its base URL and model name.

    Raises ModelValidationError, because this runs while a user is looking at a form.
    """
    raw = (reference or "").strip()
    if not raw:
        raise ModelValidationError("Give an endpoint like http://localhost:11434/v1#qwen2.5vl:7b")

    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise ModelValidationError(
            f"{raw!r} is not an http(s) endpoint; expected http://host:port/v1#model-name"
        )
    if not parts.netloc:
        raise ModelValidationError(f"{raw!r} names no host.")

    model = parts.fragment.strip()
    if not model:
        raise ModelValidationError(
            "Name the model after a '#', e.g. http://localhost:11434/v1#qwen2.5vl:7b"
        )

    base = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    return base, model


def _headers() -> dict[str, str]:
    """Request headers, carrying a bearer token only when one is configured."""
    headers = {"Content-Type": "application/json"}
    key = (os.environ.get(API_KEY_ENV) or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _clean(text: str) -> str:
    """Strip what a chatty model wraps around a transcription.

    Models fenced their answers, quoted them, or prefixed 'The text reads:' often enough that
    leaving it to the user was not defensible. This removes only wrappers -- it never edits the
    transcription itself.
    """
    out = (text or "").strip()
    fence = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", out, re.S)
    if fence:
        out = fence.group(1).strip()
    out = re.sub(r"^(?:the (?:text|line|image) (?:reads|says)|transcription)\s*[:\-]\s*", "",
                 out, flags=re.I)
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1]
    return " ".join(out.split())


class OpenAIAPICompatibleEngine(BaseEngine):
    spec = EngineSpec(
        name="openai-api-compatible",
        label="OpenAI-compatible API",
        can_segment=False,
        can_recognize=True,
        can_train=False,
        model_sources=(SOURCE_ENDPOINT,),
        file_extensions=(),        # nothing to upload: the model lives behind the endpoint
        has_default_segmenter=False,
        # no preferred_queue: the computation happens on the far side, so this engine has no claim
        # on a local GPU and must not be billed as if it had.
    )

    # --- model management

    def validate_model(self, path: str) -> ModelInfo:
        raise ModelValidationError("This engine takes an endpoint, not an uploaded file.")

    def validate_reference(self, reference: str) -> ModelInfo:
        """Check that the endpoint answers and actually serves the named model."""
        base, model = parse_reference(reference)
        served = self._list_models(base)
        if served and model not in served:
            shown = ", ".join(sorted(served)[:8])
            raise ModelValidationError(
                f"{base} does not serve {model!r}. Available there: {shown}"
            )
        return ModelInfo(job="recognition", reference=f"{base}#{model}")

    @staticmethod
    def _list_models(base: str) -> set[str]:
        """Model names the endpoint reports, or an empty set if it does not support listing.

        An endpoint that refuses to list its models is not an error: some gateways do not implement
        /models. Only an unreachable endpoint is.
        """
        import requests

        try:
            response = requests.get(f"{base}/models", headers=_headers(), timeout=15)
        except requests.RequestException as exc:
            raise ModelValidationError(f"{base} is not reachable: {exc}") from None
        if response.status_code == 404:
            return set()
        if not response.ok:
            raise ModelValidationError(
                f"{base} answered {response.status_code} when asked for its models."
            )
        try:
            payload = response.json()
        except ValueError:
            raise ModelValidationError(f"{base} did not answer with JSON.") from None
        return {entry.get("id") for entry in payload.get("data", []) if entry.get("id")}

    def version_source(self, model) -> str:
        """Record the model name rather than the URL: the host is deployment detail, the model is not."""
        reference = getattr(model, "reference", "") or ""
        name = reference.partition("#")[2] or self.spec.name
        source = f"{self.spec.name}:{name}"
        return source if len(source) <= 128 else source[:125] + "..."

    # --- recognition

    def recognize(
        self, image, lines: Iterable[LineInput], *, model, options: RecognizeOptions
    ) -> Iterator[RecognitionResult]:
        base, model_name = parse_reference(getattr(model, "reference", ""))

        for line in lines:
            try:
                payload = self._encode(image, line)
            except Exception:
                # one line that cannot be cropped is skipped; the page is still worth transcribing
                logger.exception("could not prepare line %s", line.id)
                continue

            text = self._ask(base, model_name, payload)
            yield RecognitionResult(line_id=line.id, text=text, graphemes=None)

    @staticmethod
    def _encode(image, line: LineInput) -> str:
        """Crop the line and return it as a data URI."""
        crop = crop_line(image, line, mask=True)
        if max(crop.size) > MAX_EDGE:
            ratio = MAX_EDGE / max(crop.size)
            crop = crop.resize((max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))))
        buffer = io.BytesIO()
        crop.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _ask(base: str, model_name: str, data_uri: str) -> str:
        """One chat completion for one line image. Transport failures fail the whole request."""
        import requests

        body = {
            "model": model_name,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
        }

        try:
            response = requests.post(
                f"{base}/chat/completions",
                headers=_headers(),
                data=json.dumps(body),
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise EngineError(f"{base} is not reachable: {exc}") from None

        if not response.ok:
            raise EngineError(f"{base} answered {response.status_code}: {response.text[:200]}")

        try:
            choices = response.json()["choices"]
            content = choices[0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise EngineError(f"{base} returned an unexpected answer: {exc}") from None

        return _clean(content if isinstance(content, str) else "")
