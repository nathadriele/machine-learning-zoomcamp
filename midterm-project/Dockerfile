FROM python:3.13.0-bullseye

RUN pip install gunicorn pipenv markupsafe numpy scikit-learn==1.5.2 werkzeug blinker

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["*.py", "*.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]