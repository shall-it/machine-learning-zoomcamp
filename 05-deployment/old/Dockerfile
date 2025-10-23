FROM python:3.10.12-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv && \
  pipenv install --system --deploy

COPY ["predict.py", "model_C1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]