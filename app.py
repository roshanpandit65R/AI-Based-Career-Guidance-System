from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained models
models = {
    "SVM": pickle.load(open("models/svm_model.pkl", "rb")),
    "Decision Tree (Entropy)": pickle.load(open("models/dt_entropy_model.pkl", "rb")),
    "Decision Tree (Gini)": pickle.load(open("models/dt_gini_model.pkl", "rb")),
    "Random Forest": pickle.load(open("models/rf_model.pkl", "rb"))
}

# Load dataset to get feature names
df = pd.read_csv("dataset.csv")
features = list(df.columns[:-1])  # Extract feature names

# Load label encoders
label_encoders = pickle.load(open("models/label_encoders.pkl", "rb"))

# Load model accuracy scores
try:
    with open("models/accuracy_scores.pkl", "rb") as f:
        model_accuracies = pickle.load(f)
except FileNotFoundError:
    model_accuracies = {name: 0 for name in models}  # Default if accuracy not found

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = []
        warnings = []

        for feature in features:
            value = request.form[feature]

            # Convert categorical inputs using label encoders
            if feature in label_encoders:
                if value in label_encoders[feature].classes_:
                    value = label_encoders[feature].transform([value])[0]
                else:
                    warnings.append(f"'{value}' is an unseen category for {feature}. Assigning default value 0.")
                    value = 0  # Assign default for unseen category
            else:
                try:
                    value = float(value)  # Convert numeric values
                except ValueError:
                    warnings.append(f"Invalid value for {feature}. Assigning default value 0.")
                    value = 0
        
            user_input.append(value)

        # Convert to numpy array
        user_input = np.array(user_input).reshape(1, -1)

        # Store predictions and their confidence scores
        predictions = {}
        prediction_confidences = {}

        for name, model in models.items():
            prediction = model.predict(user_input)[0]

            # Decode categorical target if needed
            target_column = df.columns[-1]
            if target_column in label_encoders:
                prediction = label_encoders[target_column].inverse_transform([prediction])[0]

            predictions[name] = prediction

            # Get prediction probability if the model supports it
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(user_input)[0]  # Get probability scores
                confidence = max(probabilities) * 100  # Get highest probability as confidence
                prediction_confidences[name] = round(confidence, 2)
            else:
                prediction_confidences[name] = 0  # Ensure numeric value

        # Determine the best model based on highest confidence
        best_model = max(prediction_confidences, key=prediction_confidences.get, default=None)
        best_prediction = predictions.get(best_model, "N/A")
        best_confidence = prediction_confidences.get(best_model, 0)

        # Get career range (top predictions)
        career_range = list(set(predictions.values()))

        return render_template(
            "result.html",
            predictions=predictions,
            prediction_confidences=prediction_confidences,  # Pass confidence scores
            best_model=best_model,
            best_prediction=best_prediction,
            best_confidence=best_confidence,
            career_range=career_range,
            model_accuracies=model_accuracies,
            warnings=warnings
        )

    return render_template("index.html", features=features)

if __name__ == "__main__":
    app.run(debug=True)