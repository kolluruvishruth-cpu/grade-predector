# Student Performance Predictor

A complete Streamlit web application that predicts a student's final grade using a Random Forest Regression model.

## Project Structure

```text
student-performance-predictor/
│
├── app.py
├── train_model.py
├── student_data.csv
├── model.pkl
├── requirements.txt
└── README.md
```

## Features

- Streamlit frontend with professional sidebar navigation
- Random Forest Regression model for grade prediction
- Joblib-based model persistence
- Interactive prediction charts
- Data visualization dashboard
- Insights page for quick analysis

## Dataset Columns

- `attendance`
- `study_hours`
- `previous_marks`
- `assignments_completed`
- `sleep_hours`
- `final_grade`

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:

   ```bash
   python train_model.py
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## Notes

- The dataset is included as `student_data.csv`.
- Running `train_model.py` creates `model.pkl`.
- If `model.pkl` is missing, `app.py` will train a fresh model automatically on first launch and save it.
