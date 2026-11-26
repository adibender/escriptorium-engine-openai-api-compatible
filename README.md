# escriptorium-engine-openai-api-compatible

Any model served behind an OpenAI-compatible chat API, as an eScriptorium engine -- in practice a
vision-language model, since the engine sends it line images. Installing this
package is the entire integration — the `escriptorium.engines` entry point in `pyproject.toml` makes
the engine appear. Nothing in eScriptorium is edited.

The model is neither a file nor a repository: it is an **endpoint** plus the name of a model served
there. The interface is the one OpenAI defined for chat completions
([reference](https://developers.openai.com/api/reference/chat-completions/overview),
[machine-readable specification](https://github.com/openai/openai-openapi)), which many other servers
implement: [Ollama](https://docs.ollama.com/api/openai-compatibility),
[vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/), LM Studio, llama.cpp.

A local server usually needs no credentials, which is the case this was built and measured against.
For an endpoint that does require a key, set `ESCRIPTORIUM_OPENAI_API_KEY` in the environment of the
worker containers and it is sent as a bearer token. The key deliberately does not live in the model
reference: that string is stored in the database and shown in the interface.

## Install

This is a plugin: it installs *into* an eScriptorium environment. Unlike an engine that runs a model
locally, it brings nothing heavy — the computation happens on the far side of an HTTP call.

### Expected in the environment

pip installs `requests` and Pillow, both of which eScriptorium already has. Everything else has to be
there already:

- eScriptorium **with the engine layer**, i.e. the `feature/engine-layer-upstream` branch of
  [gitlab.com/adibender-gitlab/escriptorium](https://gitlab.com/adibender-gitlab/escriptorium/-/tree/feature/engine-layer-upstream).
  That is where the contract this package imports (`engines.base`) lives; upstream eScriptorium does
  not read the `escriptorium.engines` entry point group yet, so the engine would install there but
  never appear. The same applies to the image you build on: use one built from that branch.
- Python ≥ 3.10.
- an **OpenAI-compatible server** (Ollama, vLLM, LM Studio, llama.cpp, …) serving a vision-language
  model, **reachable from the worker containers** (see below).
- for endpoints that require a key: `ESCRIPTORIUM_OPENAI_API_KEY` in the environment of the worker
  containers.

### With Docker (recommended)

```sh
# BASE_IMAGE defaults to escriptorium-engines:latest, so one image can carry several engines.
docker build --build-arg BASE_IMAGE=<an eScriptorium image with the engine layer> \
             -t escriptorium-engines:latest .
```

Then point the app containers at the image.

### With pip, from GitHub

Inside an eScriptorium environment that already has everything listed above:

```sh
pip install "git+https://github.com/adibender/escriptorium-engine-openai-api-compatible"
pip check
```

Append `@<commit>` or `@<tag>` to the URL to pin a version.

From a local checkout, `pip install -e .` does the same.

### Restart

**Restart every app container after installing**, including the celery workers (`web`,
`celery-main`, `celery-gpu`, `celery-intensive-inference`). Entry points are read when a process
starts, so a container that is already running never notices a newly installed engine. The same
holds after upgrading or uninstalling the package.

### Verify

`GET /api/engines/` lists `openai-api-compatible`, and *OpenAI-compatible API* appears in the
Transcribe dialog.

### Reachability

**The endpoint must be reachable from the worker containers**, which is not the same as being
reachable from the host. Ollama, for instance, listens on `127.0.0.1` by default, so a container on
a bridge network cannot see it; either bind the server to an address the containers can reach, or
run the model where they can.

## Use

1. **Add a model** with a reference of the form `http://host:port/v1#model-name`, e.g.
   `http://ollama:11434/v1#qwen2.5vl:7b`. The engine checks that the endpoint answers and that it
   actually serves that model, and stores the normalised reference.
2. **Segment the pages first**, with kraken or tesseract — this engine recognises lines and has no
   segmenter of its own.
3. **Transcribe**, choosing the model. The model carries its engine, so nothing else has to be
   selected.

Each line is cropped, masked and sent as its own request. Batching would be faster, but a model
answering about several images at once routinely mixes them up, and a silently mismatched line is
worse than a slow one.

## What it does and does not do

- **recognises lines only.** Segment with another engine first.
- reports no character positions and no confidences: `graphemes` is `None`
- **asks for no GPU queue.** The computation happens on the far side, so this engine has no claim on
  a local GPU and is not billed as if it had.
- cannot train
- strips what a chat model wraps around an answer — code fences, quotation marks, "The text reads:" —
  but never edits the transcription itself
- fails the whole request with `EngineError` when the endpoint is unreachable or answers something
  unexpected, rather than returning empty transcriptions

## How well does it read?

One measurement, on four lines of a Fraktur page (*Reichsanzeiger*) already transcribed by kraken,
against `qwen2.5vl:7b` served by Ollama on an RTX A6000:

| | kraken | qwen2.5vl:7b |
|---|---|---|
| `224` | `224` | same |
| `Verzeichniss` | `Verzeichniss` | same |
| line 3 | `Staats-Obligat**l**onen` | `Staats-Obligat**i**onen` — the model is right |
| line 4 | `Staats-An⸗` | `Staats-An**z**` — kraken is right; the model invented a letter for the hyphenation mark |

7.0 seconds for four lines, i.e. roughly two minutes for an 85-line page.

Four lines of one page is not a benchmark, and this is deliberately not presented as one. What it
does show is that the error profiles differ: the model resolves a long-s confusion kraken gets wrong,
and mishandles a historical hyphenation mark kraken gets right. That is the argument for having more
than one engine, not an argument that either is better.

## Tests

```sh
python manage.py test escriptorium_engine_openai_api_compatible    # inside an eScriptorium container
```

Nothing in the suite talks to a real endpoint: the transport is stubbed, so it runs anywhere. It
covers the conformance contract, reference parsing, the answer cleaning, and that transport failures
raise `EngineError` instead of yielding empty text.

## Relationship to eScriptorium

This is a plugin, not a standalone library: it imports the contract from `engines.base` and installs
into an eScriptorium environment, which is also the only place it runs. eScriptorium is deployed as
an application rather than published as a distribution, so there is no version for `pyproject.toml`
to depend on.

One limitation worth knowing before you build on this: eScriptorium's model form currently offers
Hugging Face references only. Until it offers endpoints too, a model for this engine has to be
created through the API or the Django shell.
