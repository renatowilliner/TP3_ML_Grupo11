"""
Módulo principal para entrenamiento de modelos de predicción de precios de Bitcoin
Basado en el análisis del EDA.ipynb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
from datetime import datetime

# Importar módulos locales
from data import load_bitcoin_data
from preprocess import clean_data, winsorize_data, create_features, prepare_training_data

def train_models(X_train, y_train, X_test, y_test):
    """
    Entrena múltiples modelos y compara su rendimiento
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
    
    Returns:
        dict: Diccionario con modelos entrenados y métricas
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    results = {}
    trained_models = {}
    
    print("Entrenando modelos...")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"Entrenando {name}...")
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_rmse': cv_rmse,
            'predictions_test': y_pred_test
        }
        
        trained_models[name] = model
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: {np.sqrt(test_mse):.2f}")
        print(f"  CV RMSE: {cv_rmse:.2f}")
        print("-" * 30)
    
    return results, trained_models

def hyperparameter_tuning(X_train, y_train, model_name='Random Forest'):
    """
    Realiza búsqueda de hiperparámetros para el mejor modelo
    
    Args:
        X_train, y_train: Datos de entrenamiento
        model_name (str): Nombre del modelo a optimizar
    
    Returns:
        dict: Mejores parámetros encontrados
    """
    print(f"Optimizando hiperparámetros para {model_name}...")
    
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
    
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = GradientBoostingRegressor(random_state=42)
    
    else:
        print(f"Optimización no implementada para {model_name}")
        return None
    
    # Búsqueda de cuadrícula
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score: {np.sqrt(-grid_search.best_score_):.2f}")
    
    return grid_search.best_params_

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evalúa un modelo y genera visualizaciones
    
    Args:
        model: Modelo entrenado
        X_test, y_test: Datos de prueba
        model_name (str): Nombre del modelo
    """
    y_pred = model.predict(X_test)
    
    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nEvaluación de {model_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Visualización
    plt.figure(figsize=(12, 5))
    
    # Gráfico 1: Predicciones vs Valores reales
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'{model_name} - Predicciones vs Reales')
    
    # Gráfico 2: Serie temporal
    plt.subplot(1, 2, 2)
    plt.plot(y_test.index, y_test.values, label='Real', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='Predicción', alpha=0.7)
    plt.xlabel('Fecha')
    plt.ylabel('Precio BTC')
    plt.title(f'{model_name} - Serie Temporal')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'predictions': y_pred
    }

def save_model(model, scaler, model_name, results):
    """
    Guarda el modelo y el scaler entrenados
    
    Args:
        model: Modelo entrenado
        scaler: Scaler entrenado
        model_name (str): Nombre del modelo
        results (dict): Resultados del modelo
    """
    # Crear directorio de modelos si no existe
    os.makedirs("../models", exist_ok=True)
    
    # Guardar modelo
    model_path = f"../models/{model_name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_path)
    
    # Guardar scaler
    scaler_path = f"../models/{model_name.lower().replace(' ', '_')}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Guardar resultados
    results_path = f"../models/{model_name.lower().replace(' ', '_')}_results.pkl"
    joblib.dump(results, results_path)
    
    print(f"Modelo guardado en: {model_path}")
    print(f"Scaler guardado en: {scaler_path}")
    print(f"Resultados guardados en: {results_path}")

def main():
    """
    Función principal para entrenar modelos
    """
    print("=== ENTRENAMIENTO DE MODELOS DE PREDICCIÓN DE BITCOIN ===")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. Cargar datos
    print("1. Cargando datos...")
    df = load_bitcoin_data()
    print(f"   Datos cargados: {df.shape}")
    
    # 2. Limpiar datos
    print("\n2. Limpiando datos...")
    df_clean = clean_data(df)
    print(f"   Datos limpios: {df_clean.shape}")
    
    # 3. Winsorizar outliers
    print("\n3. Tratando outliers...")
    df_winsor = winsorize_data(df_clean)
    
    # 4. Crear características
    print("\n4. Creando características...")
    df_features = create_features(df_winsor)
    print(f"   Características creadas: {df_features.shape}")
    
    # 5. Preparar datos para entrenamiento
    print("\n5. Preparando datos para entrenamiento...")
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(df_features)
    print(f"   Conjunto de entrenamiento: {X_train.shape}")
    print(f"   Conjunto de prueba: {X_test.shape}")
    
    # 6. Entrenar modelos
    print("\n6. Entrenando modelos...")
    results, trained_models = train_models(X_train, y_train, X_test, y_test)
    
    # 7. Seleccionar mejor modelo
    print("\n7. Seleccionando mejor modelo...")
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = trained_models[best_model_name]
    
    print(f"Mejor modelo: {best_model_name}")
    print(f"R² Score: {results[best_model_name]['test_r2']:.4f}")
    
    # 8. Optimizar hiperparámetros del mejor modelo
    print("\n8. Optimizando hiperparámetros...")
    best_params = hyperparameter_tuning(X_train, y_train, best_model_name)
    
    # 9. Evaluar modelo final
    print("\n9. Evaluando modelo final...")
    final_results = evaluate_model(best_model, X_test, y_test, best_model_name)
    
    # 10. Guardar modelo
    print("\n10. Guardando modelo...")
    save_model(best_model, scaler, best_model_name, final_results)
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    return best_model, scaler, final_results

if __name__ == "__main__":
    # Ejecutar entrenamiento
    model, scaler, results = main()
