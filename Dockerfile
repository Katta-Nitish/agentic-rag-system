FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential \
    && pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE  8501

CMD ["streamlit", "run", "assignment.py", "--server.port=8501", "--server.address=0.0.0.0"]