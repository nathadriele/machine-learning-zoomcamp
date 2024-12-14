FROM python:3.13.0-bullseye

WORKDIR /.

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system

COPY . .

CMD ["streamlit", "run", "predict.py", "--server.port=8501", "--server.address=0.0.0.0"]