FROM python:3.10.6-buster

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY howhappyineurope howhappyineurope
COPY data data
COPY setup.py setup.py
COPY Makefile Makefile
COPY README.md README.md

RUN pip install .

CMD uvicorn howhappyineurope.api.fast:app --host 0.0.0.0 --port $PORT
