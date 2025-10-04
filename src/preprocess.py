"""
Módulo para preprocesamiento de datos de Bitcoin
Basado en el análisis del EDA.ipynb
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Limpia los datos basándose en el análisis del EDA
    
    Args:
        df (pd.DataFrame): DataFrame con datos de Bitcoin
    
    Returns:
        pd.DataFrame: DataFrame limpio
    """
    df_clean = df.copy()
    
    # Corregir MultiIndex en columnas
    df_clean.columns = [col[0] if isinstance(col, tuple) else col for col in df_clean.columns]
    
    # Convertir columnas a numérico
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_clean.columns:
            df_clean.loc[:, col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Eliminar filas con valores faltantes
    df_clean = df_clean.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

def detect_outliers_iqr(df, col):
    """
    Detecta outliers usando el método IQR
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        col (str): Nombre de la columna
    
    Returns:
        pd.DataFrame: Filas que son outliers
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)]

def winsorize_data(df, columns=None, lower_percentile=0.01, upper_percentile=0.99):
    """
    Aplica winsorizing para limitar valores extremos
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columns (list): Lista de columnas a winsorizar
        lower_percentile (float): Percentil inferior
        upper_percentile (float): Percentil superior
    
    Returns:
        pd.DataFrame: DataFrame con datos winsorizados
    """
    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    df_winsor = df.copy()
    
    for col in columns:
        if col in df_winsor.columns:
            lower = df_winsor[col].quantile(lower_percentile)
            upper = df_winsor[col].quantile(upper_percentile)
            df_winsor[col] = df_winsor[col].clip(lower, upper)
            print(f"{col}: recortado a [{lower:.2f}, {upper:.2f}]")
    
    return df_winsor

def create_features(df):
    """
    Crea características adicionales para el modelo
    
    Args:
        df (pd.DataFrame): DataFrame con datos de Bitcoin
    
    Returns:
        pd.DataFrame: DataFrame con características adicionales
    """
    df_features = df.copy()
    
    # Retornos logarítmicos
    df_features['log_return'] = np.log(df_features['Close'] / df_features['Close'].shift(1))
    
    # Medias móviles
    df_features['MA_7'] = df_features['Close'].rolling(window=7).mean()
    df_features['MA_30'] = df_features['Close'].rolling(window=30).mean()
    
    # Volatilidad (desviación estándar de retornos)
    df_features['volatility_7'] = df_features['log_return'].rolling(window=7).std()
    df_features['volatility_30'] = df_features['log_return'].rolling(window=30).std()
    
    # Rango de precios
    df_features['price_range'] = df_features['High'] - df_features['Low']
    df_features['price_range_pct'] = df_features['price_range'] / df_features['Close']
    
    # Volumen relativo
    df_features['volume_ma_30'] = df_features['Volume'].rolling(window=30).mean()
    df_features['volume_ratio'] = df_features['Volume'] / df_features['volume_ma_30']
    
    return df_features

def scale_data(df, columns=None, scaler=None):
    """
    Escala los datos usando RobustScaler
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columns (list): Lista de columnas a escalar
        scaler: Scaler ya entrenado (opcional)
    
    Returns:
        tuple: (DataFrame escalado, scaler entrenado)
    """
    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Filtrar columnas que existen
    existing_columns = [col for col in columns if col in df.columns]
    
    if scaler is None:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(df[existing_columns])
    else:
        X_scaled = scaler.transform(df[existing_columns])
    
    df_scaled = pd.DataFrame(X_scaled, columns=existing_columns, index=df.index)
    
    return df_scaled, scaler

def prepare_training_data(df, target_column='Close', test_size=0.2, random_state=42):
    """
    Prepara los datos para entrenamiento
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        target_column (str): Columna objetivo
        test_size (float): Proporción de datos para test
        random_state (int): Semilla para reproducibilidad
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Crear características
    df_features = create_features(df)
    
    # Eliminar filas con NaN (por las medias móviles)
    df_features = df_features.dropna()
    
    # Definir características y objetivo
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'log_return', 
                      'MA_7', 'MA_30', 'volatility_7', 'volatility_30',
                      'price_range', 'price_range_pct', 'volume_ratio']
    
    # Filtrar columnas que existen
    existing_features = [col for col in feature_columns if col in df_features.columns]
    
    X = df_features[existing_features]
    y = df_features[target_column]
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    # Escalar características
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de vuelta a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=existing_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=existing_features, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Ejemplo de uso
    from data import load_bitcoin_data
    
    # Cargar datos
    df = load_bitcoin_data()
    
    # Limpiar datos
    df_clean = clean_data(df)
    print(f"Datos limpios: {df_clean.shape}")
    
    # Winsorizar
    df_winsor = winsorize_data(df_clean)
    
    # Crear características
    df_features = create_features(df_winsor)
    print(f"Características creadas: {df_features.shape}")
    
    # Preparar datos para entrenamiento
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(df_features)
    print(f"Conjunto de entrenamiento: {X_train.shape}")
    print(f"Conjunto de prueba: {X_test.shape}")
