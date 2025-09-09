from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Load dataset
csv_path = "StudentPerformanceFactors.csv"
df = pd.read_csv(csv_path)

# Features and target
feature_cols = [
	"Hours_Studied",
	"Attendance",
	"Sleep_Hours",
	"Previous_Scores",
	"Tutoring_Sessions",
]
X = df[feature_cols]
y = df["Exam_Score"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Linear Regression with scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin = LinearRegression()
lin.fit(X_train_scaled, y_train)
y_pred_lin = lin.predict(X_test_scaled)

# Polynomial Regression (degree=2) with scaling inside a pipeline
poly2 = Pipeline([
	("scaler", StandardScaler()),
	("poly", PolynomialFeatures(degree=2, include_bias=False)),
	("lin", LinearRegression()),
])
poly2.fit(X_train, y_train)
y_pred_poly2 = poly2.predict(X_test)

# Example predictions first (5 random rows from test set)
import numpy as np
np.random.seed(42)  # For reproducible random selection
show_n = min(5, len(y_test))
random_indices = np.random.choice(len(y_test), size=show_n, replace=False)
print("Example predictions (5 random rows from test set):")
for i, idx in enumerate(random_indices):
	print(
		f"{i+1}) Actual: {y_test.iloc[idx]:.1f} | Linear: {y_pred_lin[idx]:.1f} | Polynomial(d=2): {y_pred_poly2[idx]:.1f}"
	)

# Then metrics
print("\n[Linear Regression]")
print(f"MSE: {mean_squared_error(y_test, y_pred_lin):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lin):.4f}")
print(f"R^2: {r2_score(y_test, y_pred_lin):.4f}")

print("\n[Polynomial Regression (degree=2)]")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly2):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_poly2):.4f}")
print(f"R^2: {r2_score(y_test, y_pred_poly2):.4f}")

# Quick comparison
r2_lin = r2_score(y_test, y_pred_lin)
r2_poly2 = r2_score(y_test, y_pred_poly2)
best = "Linear Regression" if r2_lin >= r2_poly2 else "Polynomial Regression (degree=2)"
print(f"\nBest by R^2: {best}")