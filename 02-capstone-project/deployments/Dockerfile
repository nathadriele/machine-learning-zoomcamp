FROM python:3.9-slim-buster

RUN pip install pipenv

WORKDIR /app  
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["*.py", "uber_price_prediction.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]