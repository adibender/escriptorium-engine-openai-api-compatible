# How a deployment adds this engine: extend an eScriptorium image and pip install the package.
# Layered on the image that already carries the other engines so one image holds all of them; any
# eScriptorium image works as the base. Nothing in eScriptorium changes.
ARG BASE_IMAGE=escriptorium-engines:latest
FROM ${BASE_IMAGE}

COPY . /opt/escriptorium-engine-openai-api-compatible

# Unlike the TrOCR package this brings nothing heavy: the model runs on the far side of an HTTP
# endpoint, so the dependencies are requests and Pillow, both of which eScriptorium already has.
# pip check still runs, because "adds almost nothing" is a claim worth verifying rather than
# assuming.
RUN pip install --no-cache-dir -e /opt/escriptorium-engine-openai-api-compatible \
    && pip check
