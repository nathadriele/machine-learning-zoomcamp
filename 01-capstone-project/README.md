## Ford Car Price Prediction

### Overview
This project predicts the price of Ford cars based on features such as model, year, transmission, mileage, fuel type, tax, mpg, and engine size.

### Steps
- EDA available in: `01-capstone-project.ipynb`.
- Model training: `python train.py`
- Model testing: `python test.py`
- Application via Streamlit: `streamlit run predict.py`

### How to Runs
1. Install dependencies: `pip install -r requirements.txt` or use Pipenv: `pipenv install --deploy --system`.
2. Run training: `python train.py`
3. Run testing: `python test.py`
4. Run the web application: `streamlit run predict.py` and access `http://localhost:8501`.

### Docker
- Build the image: `docker build -t ford-car-price .`
- Run the container: `docker run -p 8501:8501 ford-car-price`
