"""
Módulo para generar predicciones usando modelos entrenados
"""
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import yfinance as yf

from data import load_bitcoin_data
from preprocess import clean_data, winsorize_data, create_features, scale_data

def load_trained_model(model_name="random_forest"):
    """
    Carga un modelo entrenado y su scaler
    
    Args:
        model_name (str): Nombre del modelo a cargar
    
    Returns:
        tuple: (modelo, scaler)
    """
    model_path = f"../models/{model_name}_model.pkl"
    scaler_path = f"../models/{model_name}_scaler.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Modelo {model_name} cargado exitosamente")
    return model, scaler

def prepare_prediction_data(df, scaler):
    """
    Prepara los datos para predicción usando el mismo preprocesamiento
    
    Args:
        df (pd.DataFrame): DataFrame con datos de Bitcoin
        scaler: Scaler entrenado
    
    Returns:
        pd.DataFrame: Datos preparados para predicción
    """
    # Limpiar datos
    df_clean = clean_data(df)
    
    # Winsorizar
    df_winsor = winsorize_data(df_clean)
    
    # Crear características
    df_features = create_features(df_winsor)
    
    # Eliminar filas con NaN
    df_features = df_features.dropna()
    
    # Definir características (debe coincidir con el entrenamiento)
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'log_return', 
                      'MA_7', 'MA_30', 'volatility_7', 'volatility_30',
                      'price_range', 'price_range_pct', 'volume_ratio']
    
    # Filtrar columnas que existen
    existing_features = [col for col in feature_columns if col in df_features.columns]
    
    # Seleccionar características
    X = df_features[existing_features]
    
    # Escalar usando el scaler entrenado
    X_scaled, _ = scale_data(df_features, existing_features, scaler)
    
    return X_scaled, df_features

def predict_next_trading_day(model, scaler, days_back=30):
    """
    Predice el precio para el próximo día de trading usando los últimos N días
    
    Args:
        model: Modelo entrenado
        scaler: Scaler entrenado
        days_back (int): Número de días hacia atrás para usar en la predicción
    
    Returns:
        dict: Diccionario con la predicción y metadatos
    """
    # Cargar datos recientes (hasta hoy)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days_back + 10)).strftime('%Y-%m-%d')
    
    df = load_bitcoin_data(start_date=start_date, end_date=end_date, save_to_csv=False)
    
    # Preparar datos
    X_scaled, df_features = prepare_prediction_data(df, scaler)
    
    # Usar los últimos datos disponibles para la predicción
    X_latest = X_scaled.tail(1)
    
    # Hacer predicción
    prediction = model.predict(X_latest)[0]
    
    # Obtener información del último día disponible
    last_date = df_features.index[-1]
    last_price = df_features['Close'].iloc[-1]
    
    # Calcular el próximo día de trading (día siguiente)
    next_trading_day = last_date + timedelta(days=7)
    
    # Calcular cambio porcentual
    price_change = ((prediction - last_price) / last_price) * 100
    
    result = {
        'target_date': next_trading_day.strftime('%Y-%m-%d'),
        'predicted_price': prediction,
        'last_known_date': last_date,
        'last_known_price': last_price,
        'price_change_usd': prediction - last_price,
        'price_change_percent': price_change,
        'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return result

def predict_next_day(model, scaler, days_back=30):
    """
    Predice el precio del próximo día usando los últimos N días
    
    Args:
        model: Modelo entrenado
        scaler: Scaler entrenado
        days_back (int): Número de días hacia atrás para usar en la predicción
    
    Returns:
        dict: Diccionario con la predicción y metadatos
    """
    # Cargar datos recientes
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days_back + 10)).strftime('%Y-%m-%d')
    
    df = load_bitcoin_data(start_date=start_date, end_date=end_date, save_to_csv=False)
    
    # Preparar datos
    X_scaled, df_features = prepare_prediction_data(df, scaler)
    
    # Usar los últimos datos disponibles
    X_latest = X_scaled.tail(1)
    
    # Hacer predicción
    prediction = model.predict(X_latest)[0]
    
    # Obtener información del último día
    last_date = df_features.index[-1]
    last_price = df_features['Close'].iloc[-1]
    
    # Calcular cambio porcentual
    price_change = ((prediction - last_price) / last_price) * 100
    
    result = {
        'prediction_date': last_date,
        'predicted_price': prediction,
        'last_known_price': last_price,
        'price_change_usd': prediction - last_price,
        'price_change_percent': price_change,
        'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return result

def predict_multiple_days(model, scaler, n_days=7, days_back=30):
    """
    Predice los precios para los próximos N días
    
    Args:
        model: Modelo entrenado
        scaler: Scaler entrenado
        n_days (int): Número de días a predecir
        days_back (int): Número de días hacia atrás para usar
    
    Returns:
        pd.DataFrame: DataFrame con predicciones
    """
    predictions = []
    
    for day in range(1, n_days + 1):
        # Para predicciones múltiples, usaríamos un enfoque más sofisticado
        # Por simplicidad, usamos la misma predicción (en la práctica se usaría un modelo de series temporales)
        result = predict_next_day(model, scaler, days_back)
        result['prediction_day'] = day
        predictions.append(result)
    
    return pd.DataFrame(predictions)

def generate_submission_file(model_name="linear_regression", output_file="../submission.csv"):
    """
    Genera un archivo de submission con predicciones para el próximo día de trading
    
    Args:
        model_name (str): Nombre del modelo a usar
        output_file (str): Ruta del archivo de salida
    """
    try:
        # Cargar modelo
        model, scaler = load_trained_model(model_name)
        
        # Generar predicción para el próximo día de trading
        prediction = predict_next_trading_day(model, scaler)
        
        # Crear DataFrame de submission
        submission_df = pd.DataFrame([{
            'target_date': prediction['target_date'],
            'predicted_price': prediction['predicted_price'],
            'last_known_date': prediction['last_known_date'],
            'last_known_price': prediction['last_known_price'],
            'price_change_percent': prediction['price_change_percent']
        }])
        
        # Guardar archivo
        submission_df.to_csv(output_file, index=False)
        
        print(f"Archivo de submission generado: {output_file}")
        print(f"Predicción para {prediction['target_date']}: ${prediction['predicted_price']:.2f}")
        print(f"Basado en datos hasta: {prediction['last_known_date']}")
        print(f"Último precio conocido: ${prediction['last_known_price']:.2f}")
        print(f"Cambio predicho: {prediction['price_change_percent']:.2f}%")
        
        return submission_df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrate de haber entrenado el modelo primero ejecutando train.py")
        return None

def main():
    """
    Función principal para generar predicciones
    """
    print("=== GENERACIÓN DE PREDICCIONES ===")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)
    
    # Generar archivo de submission
    submission = generate_submission_file()
    
    if submission is not None:
        print("\nPredicción generada exitosamente:")
        print(submission)
    
    return submission

if __name__ == "__main__":
    # Ejecutar predicción
    submission = main()
