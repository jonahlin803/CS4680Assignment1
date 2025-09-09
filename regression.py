from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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

print("[Linear Regression]")
print(f"MSE: {mean_squared_error(y_test, y_pred_lin):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lin):.4f}")
print(f"R^2: {r2_score(y_test, y_pred_lin):.4f}")

# Random Forest Regression (no scaling needed)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n[Random Forest Regression]")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")
print(f"R^2: {r2_score(y_test, y_pred_rf):.4f}")

# Quick comparison
r2_lin = r2_score(y_test, y_pred_lin)
r2_rf = r2_score(y_test, y_pred_rf)
best = "Linear Regression" if r2_lin >= r2_rf else "Random Forest Regression"
print(f"\nBest by R^2: {best}")

# Example predictions (first 5 rows)
show_n = min(5, len(y_test))
print("\nExample predictions (first 5 rows of test set):")
for i in range(show_n):
	print(
		f"{i+1}) Actual: {y_test.iloc[i]:.1f} | LinReg: {y_pred_lin[i]:.1f} | RF: {y_pred_rf[i]:.1f}"
	)
