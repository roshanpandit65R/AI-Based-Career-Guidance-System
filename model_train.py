import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert categorical data to numeric using Label Encoding


label_encoders = {}  # Store encoders for each categorical column

for column in df.columns:
    if df[column].dtype == 'object':  # If column contains strings
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Store encoder for later use

# Save label encoders
if not os.path.exists("models"):
    os.makedirs("models")

pickle.dump(label_encoders, open("models/label_encoders.pkl", "wb"))

# Separate features (X) and target (y)
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target (Career field)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save models
models = {
    "SVM": SVC(),
    "Decision Tree (Entropy)": DecisionTreeClassifier(criterion="entropy"),
    "Decision Tree (Gini)": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

accuracy_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pickle.dump(model, open(f"models/{name.lower().replace(' ', '_')}_model.pkl", "wb"))

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy_scores[name] = accuracy_score(y_test, y_pred)

# Save accuracy scores for later use in Flask app
pickle.dump(accuracy_scores, open("models/accuracy_scores.pkl", "wb"))

# Print accuracy scores
for model_name, acc in accuracy_scores.items():
    print(f"{model_name} Accuracy: {acc:.4f}")
