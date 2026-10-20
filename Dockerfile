# MatchSense web service: the live pipeline, the websocket stream and the browser frontend.
#
#   docker build -t matchsense:local .
#   docker run --rm -p 8000:8000 matchsense:local          # local narration, no Azure needed
#
# On Azure Container Apps the app finds Microsoft Foundry through environment variables
# (see infra/main.bicep) and signs in with its managed identity: there are no keys in the image.

ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install ".[azure]"

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    MATCHSENSE_DATA=/app/data/matches
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY data/matches /app/data/matches
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"]
LABEL org.opencontainers.image.title="matchsense" \
      org.opencontainers.image.description="Explainable, personalised match intelligence from synthetic football data" \
      org.opencontainers.image.licenses="MIT"
CMD ["matchsense-web", "--host", "0.0.0.0", "--port", "8000"]
