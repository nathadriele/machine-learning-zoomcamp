FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"] 

RUN pipenv install --system --deploy

RUN pipenv install gunicorn

COPY Q4_score_user_web.py ./score.py

ENV MODEL_NAME=model2.bin

EXPOSE 8080

CMD [ "gunicorn", "-b", "0.0.0.0:8080", "score:app" ]