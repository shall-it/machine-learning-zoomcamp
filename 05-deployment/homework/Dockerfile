FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

COPY "predict_v3.py" ./

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict_v3:app", "--host", "0.0.0.0", "--port", "9696"]