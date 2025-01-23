install:
    pip install -r requirements.txt

run:
    python predict.py

docker-build:
    docker build -t uber-price-prediction:latest .

docker-run:
    docker run -p 5000:5000 uber-price-prediction:latest