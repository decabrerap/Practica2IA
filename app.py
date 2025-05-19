from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("wine_quality_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([
        data["fixed acidity"],
        data["volatile acidity"],
        data["citric acid"],
        data["residual sugar"],
        data["chlorides"],
        data["free sulfur dioxide"],
        data["total sulfur dioxide"],
        data["density"],
        data["pH"],
        data["sulphates"],
        data["alcohol"]
    ]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(port=5000)
