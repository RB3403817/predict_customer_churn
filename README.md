# Customer Churn Prediction with Boosting

This project provides an end-to-end solution for predicting customer churn using the Telco Customer Churn dataset. It leverages state-of-the-art boosting algorithms (XGBoost, LightGBM, CatBoost, AdaBoost) and includes data preprocessing, feature engineering, model tuning, evaluation, and explainability.

## Project Structure

```
predict_customer_churn/
├── src/
│   └── predict_customer_churn/
│       ├── __init__.py           # Package initialization
│       ├── config.py              # Configuration and constants
│       ├── data.py                # Data loading and preprocessing
│       ├── preprocessing.py       # Feature preprocessing pipelines
│       ├── models.py              # Model training and tuning
│       └── evaluation.py          # Model evaluation and visualization
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
├── churn_boosting_notebook.ipynb # Original notebook (for reference)
├── main.py                        # Main script to run the pipeline
├── pyproject.toml                 # Project dependencies
└── README.md                      # This file
```

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

### Using the Script (Recommended)
```bash
python main.py
```

This will:
1. Load the customer churn dataset
2. Preprocess and engineer features
3. Train baseline models (XGBoost, LightGBM, CatBoost, AdaBoost)
4. Display performance metrics

### Using the Notebook
1. Open `churn_boosting_notebook.ipynb`
2. Run cells sequentially for interactive exploration

### Hyperparameter Tuning
To enable hyperparameter tuning with Optuna, uncomment the tuning section in `main.py` or use the functions in `src/predict_customer_churn/models.py`.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, matplotlib

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna matplotlib
```

Or use the project dependencies:
```bash
pip install -e .
```

## Notes
- The codebase has been refactored into modular Python packages for better maintainability
- The notebook is still available for reference and interactive exploration
- Use `main.py` for production-like pipeline execution
- Model artifacts are automatically excluded from version control (see `.gitignore`)
- SHAP plots may be slow for large datasets; sampling is used for speed (see notebook)

## License
MIT License

## Acknowledgments
- [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP
