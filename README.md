# CS4680 Assignment 1 - Machine Learning Exercise

## Problem Identification
- Real-world problem: Predicting student exam scores from study/behavior factors
- Target: `Exam_Score` (continuous) - predicting exact exam performance
- Features: `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`
- Rationale: These factors directly influence student performance and are measurable predictors

## Dataset
- File: `StudentPerformanceFactors.csv`
- 6,600+ rows with student factors and exam outcomes
- Features are numeric and ready for modeling

## Models Compared
Two regression approaches to predict `Exam_Score`:
1. **Linear Regression** (with feature scaling)
2. **Polynomial Regression** (degree=2) via `PolynomialFeatures` + LinearRegression

## Code
- File: `regression.py`
- Pipeline: train/test split → scaling → fit both models → evaluate
- Output: sample predictions (first 5) → MSE/MAE/R² for each model → best by R²

## How to Run
```bash
python regression.py
```

## Report

### Executive Summary
- Goal: Compare Linear vs Polynomial regression for predicting student exam scores
- Dataset: 6,600+ student records with 5 study/behavior features
- Key finding: <fill in after running> - which model performed better and by how much

### Methods
- **Linear Regression**: Standard linear model with scaled features
- **Polynomial Regression**: Degree-2 polynomial features (interactions + squared terms) with linear regression
- Evaluation: 80/20 train/test split, metrics: MSE, MAE, R²

### Results
- Linear Regression: R² = <fill>, MSE = <fill>, MAE = <fill>
- Polynomial (degree=2): R² = <fill>, MSE = <fill>, MAE = <fill>
- **Best model**: <Linear|Polynomial> by R² = <fill>

### Discussion
- **Model comparison**: Explain which performed better and why
  - If Linear won: data relationships are approximately linear
  - If Polynomial won: nonlinear interactions between features matter
- **Feature importance**: Which features seem most predictive (can infer from coefficients)
- **Limitations**: Only 5 features used, no hyperparameter tuning, single train/test split

### Conclusion
- Successfully compared two regression approaches on real student data
- <Model name> performed better, suggesting <linear/nonlinear> relationships in the data
- Assignment requirements met: real problem, scikit-learn models, proper evaluation

## Notes
- Run `python regression.py` to see results
- Ensure `pandas` and `scikit-learn` are installed