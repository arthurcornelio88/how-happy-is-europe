FROM python:3.10.6-buster

COPY how-happy-in-europe /how-happy-in-europe
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py

RUN pip install -r requirements.txt
RUN pip install .

CMD uvicorn how-happy-in-europe.api.fast:app --host 0.0.0.0 --port $PORT
