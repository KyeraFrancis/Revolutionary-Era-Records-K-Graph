FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install -e ".[nlp,graph]" && python -m spacy download en_core_web_sm
ENTRYPOINT ["rcgraph"]
CMD ["--help"]
