
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

#Imprime la version de scikit-learn
import sklearn
print("scikit-learn version:", sklearn.__version__)

# Cargar el dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Preparar datos
X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:9090")
mlflow.set_experiment("Wine_Quality_Experiment")

# Entrenamiento
with mlflow.start_run():
    model = Ridge(alpha=0.5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Logs parametros y metricas
    mlflow.log_param("alpha", 0.5)
    mlflow.log_metric("rmse", rmse)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Guardar modelo localmente
    joblib.dump(model, "wine_quality_model.pkl")
    print(f"Modelo guardado. RMSE: {rmse}")
