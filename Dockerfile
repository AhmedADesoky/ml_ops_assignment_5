FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

RUN echo "Downloading model artifacts for RUN_ID=${RUN_ID}" \
    && echo "(Mock) Model download complete"

CMD ["python", "-c", "print('Container is ready for model serving')"]
