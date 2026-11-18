FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
# Set --build-arg BACKENDS=true to include boto3 (Amazon Bedrock).
ARG BACKENDS=false
COPY requirements.txt requirements-backends.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$BACKENDS" = "true" ]; then pip install --no-cache-dir -r requirements-backends.txt; fi

COPY assistant ./assistant
COPY knowledge ./knowledge

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["sh", "-c", "uvicorn assistant.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
