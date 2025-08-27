# Customer Churn Prediction with Boosting

This project provides an end-to-end solution for predicting customer churn using the Telco Customer Churn dataset. It leverages state-of-the-art boosting algorithms (XGBoost, LightGBM, CatBoost, AdaBoost) and includes data preprocessing, feature engineering, model tuning, evaluation, and explainability.

## Project Structure
- `churn_boosting_notebook.ipynb`: Main notebook with all steps from data loading to model explainability.
- `main.py`: (Optional) Script for running the pipeline.
- `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`: Raw dataset (add this file before running).
- `catboost_info/`: CatBoost training logs and artifacts.
- `pyproject.toml`, `uv.lock`: Dependency management files.

## Steps Covered
1. **Data Collection & Loading**: Loads the Telco Customer Churn dataset.
2. **Exploratory Data Analysis (EDA)**: Summarizes and visualizes the data.
3. **Data Preprocessing**: Cleans data, handles missing values, encodes categoricals.
4. **Feature Engineering**: Creates new features to improve model performance.
5. **Train/Test Split**: Splits data for unbiased evaluation.
6. **Modeling**: Trains XGBoost, LightGBM, CatBoost, and AdaBoost models.
7. **Hyperparameter Tuning**: Uses Optuna for automated tuning.
8. **Evaluation**: Compares models using ROC-AUC, F1, confusion matrix, and plots.
9. **Explainability**: Uses SHAP to interpret model predictions.

## Results
- AdaBoost and CatBoost performed best in initial tests, with ROC-AUC and F1 scores above 0.79.
- The tuned XGBoost model achieved a holdout ROC-AUC of ~0.85 and overall accuracy of 81%.
- SHAP analysis highlights the most important features driving churn predictions.

## How to Run
1. Place the dataset CSV in `data/raw/`.
2. Open `churn_boosting_notebook.ipynb` and run cells sequentially.
3. (Optional) Use `main.py` for a script-based workflow.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, shap, matplotlib
- Use `uv` or pip to install dependencies (see `pyproject.toml` or `requirements.txt` if available).

## Notes
- The notebook is modular: you can run only the steps you need.
- SHAP plots may be slow for large datasets; sampling is used for speed.

## License
MIT License

## Acknowledgments
- [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP
