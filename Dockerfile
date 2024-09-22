FROM python:3.12.3-slim

WORKDIR /RAG-APP


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD [ "python","app.py" ]