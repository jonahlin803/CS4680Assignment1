from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
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
y = df["Access_to_Resources"]  # categorical target: High/Medium/Low

# Train/test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM classifier
model = svm.SVC(kernel="rbf", random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n" + classification_report(y_test, y_pred))

# Example outputs: show first 5 predictions vs actual
to_show = min(5, len(y_test))
print("Example predictions (first 5):")
for i in range(to_show):
	print(f"{i+1}) Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")