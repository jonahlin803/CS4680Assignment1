# CS4680 Assignment 1 — Regression Model Comparison

## Problem Statement
The goal of this assignment is to predict **student exam performance** using regression analysis. We use the dataset `StudentPerformanceFactors.csv`, which contains features related to study habits, attendance, and prior academic history.

We implement and compare two regression models:
- **Linear Regression** (baseline)
- **Polynomial Regression (degree=2)** (to capture possible non-linear relationships)

---

## Dataset Description
- **Target Variable:**
  - `Exam_Score` → student’s final exam score (continuous, 0–100)

- **Features:**
  - `Hours_Studied` → number of hours studied per week
  - `Attendance` → class attendance percentage
  - `Sleep_Hours` → average hours of sleep per night
  - `Previous_Scores` → prior exam scores
  - `Tutoring_Sessions` → number of tutoring sessions attended

The dataset is split into **80% training** and **20% testing**.

---

## Methodology
1. **Preprocessing**
   - Features are standardized with `StandardScaler`.
   - For Linear Regression, scaling is applied directly to the features.
   - For Polynomial Regression, features are expanded into quadratic terms (`PolynomialFeatures(degree=2)`), then scaled inside a pipeline.

2. **Models**
   - **Linear Regression:** assumes a straight-line relationship between features and the target.
   - **Polynomial Regression (degree=2):** adds squared and interaction terms to allow for curved/non-linear relationships.

3. **Evaluation Metrics**
   - **MSE** (Mean Squared Error)
   - **MAE** (Mean Absolute Error)
   - **R²** (coefficient of determination)

---

## Results

Example predictions (5 random test samples):
```
1) Actual: 69.0 | Linear: 67.4 | Polynomial(d=2): 67.4
2) Actual: 68.0 | Linear: 68.4 | Polynomial(d=2): 68.5
3) Actual: 64.0 | Linear: 65.7 | Polynomial(d=2): 65.8
4) Actual: 67.0 | Linear: 66.3 | Polynomial(d=2): 66.4
5) Actual: 73.0 | Linear: 71.9 | Polynomial(d=2): 72.1
```

| Model                         | MSE    | MAE    | R²    |
|-------------------------------|--------|--------|-------|
| Linear Regression             | 5.0771 | 1.2751 | 0.6408 |
| Polynomial Regression (deg=2) | 5.1039 | 1.2796 | 0.6389 |

---

## Discussion
- **Linear Regression** produced slightly better results than Polynomial Regression.
- The two models performed nearly identically, which suggests the relationship between features and exam scores is mostly linear.
- Adding polynomial terms did not improve performance and may risk unnecessary complexity.

---

## Conclusion
- Both models are valid regression approaches.
- For this dataset, **Linear Regression performed marginally better** and is the more appropriate choice.
- Future work could include trying more flexible models like **Random Forest Regressor** or **Gradient Boosting** to explore non-linear relationships further.

---

## How to Run the Code

### Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the script
```bash
python regression.py
```

### Output
- Console will display:
  - Example predictions (5 random test samples)
  - Metrics (MSE, MAE, R²) for both models
  - A quick comparison stating which model had the better R² score
