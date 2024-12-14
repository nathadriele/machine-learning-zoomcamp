FROM python:3.13.0-bullseye

WORKDIR /.

RUN pip install pipenv 

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY . .


CMD ["streamlit", "run", "predict.py"]