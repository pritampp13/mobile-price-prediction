# Mobile Price Range Prediction

Introduction to Data Science (S2-25_DSECLZG532) — BITS Pilani WILP
Group 2 — PATIL PRITAM WAMAN (2025dc04031), AUXILEA OSANA S (2025dc04028),
ABHIMANYU SINGH (2025dc04084), DEEPAK DWIVEDI (2025dc04066)

A multi-class classification pipeline that predicts a mobile phone's price
range (0–3) from its technical specs, deployed as an interactive Streamlit app.

## Project Structure
```
.
├── Group2.ipynb                        # Full EDA + model training notebook
├── Group2.html                         # Rendered notebook output (for submission)
├── app.py                              # Streamlit deployment app
├── Mobile_Price_Prediction_train.csv
├── Mobile_Price_Prediction_test.csv
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── .gitignore
```

## Run locally (no Docker)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the notebook once to generate the .pkl model artifacts
jupyter nbconvert --to notebook --execute --inplace Group2.ipynb

streamlit run app.py
```

## Run with Docker
```bash
docker build -t mobile-price-app .
docker run -p 8501:8501 mobile-price-app
```
or, with Docker Compose:
```bash
docker compose up --build
```
Then open http://localhost:8501

The Dockerfile executes the notebook during the image build, so the model is
trained fresh inside the container — no need to commit the `.pkl` files to git.
