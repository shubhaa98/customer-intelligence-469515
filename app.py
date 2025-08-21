from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ===== Load models =====
with open("models/churn_model.pkl", "rb") as f:
    churn_model = pickle.load(f)

with open("models/segment_model.pkl", "rb") as f:
    segment_model = pickle.load(f)

with open("models/forecast_model.pkl", "rb") as f:
    forecast_model = pickle.load(f)

with open("models/sentiment_vectorizer.pkl", "rb") as f:
    sentiment_vec = pickle.load(f)

with open("models/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)


# ===== Health check =====
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Customer Intelligence API is running"}), 200


# ===== Prediction endpoint =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        model_type = data.get("model_type")
        features = data.get("features")
        text = data.get("text")
        past_sales = data.get("past_sales")

        if not model_type:
            return jsonify({"error": "model_type is required"}), 400

        # ---- Churn ----
        if model_type == "churn":
            if not features:
                return jsonify({"error": "features are required"}), 400
            X = np.array(features).reshape(1, -1)
            prediction = churn_model.predict(X)[0]
            return jsonify({"model_type": "churn", "prediction": int(prediction)})

        # ---- Segment ----
        elif model_type == "segment":
            if not features:
                return jsonify({"error": "features are required"}), 400
            X = np.array(features).reshape(1, -1)
            prediction = segment_model.predict(X)[0]
            return jsonify({"model_type": "segment", "prediction": int(prediction)})

        # ---- Forecast ----
        elif model_type == "forecast":
            if not past_sales:
                return jsonify({"error": "past_sales is required"}), 400
            X = np.array(past_sales).reshape(1, -1)
            prediction = forecast_model.predict(X)[0]
            return jsonify({"model_type": "forecast", "next_period_forecast": float(prediction)})

        # ---- Sentiment ----        
        elif model_type == "sentiment":
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                return jsonify({"error": "text is required"}), 400

            # Vectorize raw text exactly as trained in the notebook
            X_text = sentiment_vec.transform([text])
            pred = sentiment_model.predict(X_text)[0]

            # Normalize capitalization to match desired output exactly
            label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
            label = label_map.get(str(pred).strip().lower(), str(pred))

            # Return ONLY the sentiment field
            return jsonify({"sentiment": label})

        else:
            return jsonify({"error": "Invalid model_type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
