# LabelWise application image.
#
# Used by both the API and the Streamlit UI; the two services differ only in
# the command they run.

FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first so the layer caches across source edits.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev

COPY app/ ./app/
COPY ingestion/ ./ingestion/
COPY evaluation/ ./evaluation/
COPY streamlit_app/ ./streamlit_app/
COPY scripts/ ./scripts/
COPY sql/ ./sql/

EXPOSE 8000 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]