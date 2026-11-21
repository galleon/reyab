FROM python:3.12-slim
WORKDIR /app
# CPU-only torch: the classifier is a DeBERTa-base, ~50-150 ms/prompt on CPU,
# and keeping it off the GPU leaves all unified memory to the two vLLM engines.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
