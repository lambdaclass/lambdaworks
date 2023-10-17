FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends g++ libgmp3-dev

COPY requirements.txt /

RUN pip install -r /requirements.txt

ENTRYPOINT ["cairo-compile"]
