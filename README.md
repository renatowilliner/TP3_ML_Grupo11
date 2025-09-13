repo/
├─ data_raw/                  # CSV descargados desde Yahoo Finance (BTC-USD), versión cruda para reproducibilidad
├─ data_processed/            # Datos limpios, sin NaN, outliers tratados y escalados, listos para modelado
├─ analisis_exploratorios/
│   └─ EDA.ipynb              # Notebook con análisis exploratorio de datos, visualizaciones, estadísticos y outliers
├─ features/
│   └─ create_features.py     # Funciones o notebook para generar nuevas features: lag, rolling, indicadores técnicos, volumen relativo
├─ experiments/
│   ├─ baseline.ipynb         # Notebook con modelos simples de baseline (por ejemplo, predicción naive)
│   ├─ xgboost_experiment.ipynb # Experimentos con XGBoost
│   ├─ lstm_experiment.ipynb  # Experimentos con redes LSTM
│   └─ experiments_log.csv    # Registro de experimentos y resultados (alternativa: MLflow)
├─ models/
│   └─ saved_models/          # Modelos entrenados guardados (.pkl, .joblib, SavedModel)
├─ results/
│   └─ submissions/           # CSV listos para entregar/subir con predicciones finales
├─ src/
│   ├─ data.py                # Funciones para carga de datos, splits (train/test/validation)
│   ├─ preprocess.py          # Funciones de limpieza, escalado, winsorizing y pipeline reproducible
│   ├─ train.py               # Script principal para entrenar modelos
│   └─ predict_for_submission.py # Script para generar predicciones de 7 días y guardar CSV listo para entrega
├─ requirements.txt           # Librerías necesarias para correr el proyecto
├─ README.md                  # Descripción general del proyecto, estructura de repo y pasos para reproducir
└─ memoria_TP_ML_grupo_X.pdf  # Memoria técnica final (2 páginas) con decisiones, metodología y conclusiones
