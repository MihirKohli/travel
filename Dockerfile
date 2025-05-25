FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8006


CMD ["uvicorn", "main:app","--port","8006","--host","0.0.0.0"]
