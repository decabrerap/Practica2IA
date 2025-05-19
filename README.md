# Practica 2: Proyecto de Machine Learning con MLflow y Flask

Este proyecto implementa un flujo completo de Machine Learning para predecir la calidad del vino usando el siguiente dataset: `Wine Quality`. 
Se utilizó **MLflow** para gestionar experimentos y **Flask** para desplegar el modelo como un servicio web.

---

## Requisitos

- Python 3.13+
- Pip
- Entorno virtual recomendado (`venv`)
- PyCharm
- MLflow ejecutándose en el puerto 9090
- Flask

---

## Estructura del proyecto
Practica2IA/
    app.py -> API Flask que expone el modelo
    main.py -> Entrenamiento y logging del modelo
    requirements.txt -> Dependencias del proyecto
    README.md -> Guia del proyecto
    mlruns/ -> Carpeta generada por MLflow


# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt