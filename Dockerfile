FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Download the requirements 
RUN pip install -r requirements.txt

COPY rsc /app/rsc

RUN python -m rsc.train

CMD ["python", "rsc/predict.py"]