"""
Módulo para carga y manejo de datos de Bitcoin
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
import os

def load_bitcoin_data(start_date="2020-01-01", end_date=None, save_to_csv=True):
    """
    Carga datos históricos de Bitcoin desde Yahoo Finance
    
    Args:
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'. Si es None, usa la fecha actual
        save_to_csv (bool): Si guardar los datos en CSV
    
    Returns:
        pd.DataFrame: DataFrame con los datos de Bitcoin
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Cargando datos de Bitcoin desde {start_date} hasta {end_date}")
    
    # Descargar datos
    df = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")
    
    # Guardar en CSV si se solicita
    if save_to_csv:
        os.makedirs("../data_raw", exist_ok=True)
        df.to_csv("../data_raw/btc_usd_raw.csv")
        print("Datos guardados en ../data_raw/btc_usd_raw.csv")
    
    return df

def get_data_info(df):
    """
    Obtiene información básica del DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
    
    Returns:
        dict: Diccionario con información del DataFrame
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'date_range': (df.index.min(), df.index.max()),
        'missing_values': df.isna().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    return info

if __name__ == "__main__":
    # Ejemplo de uso
    df = load_bitcoin_data()
    print("Información de los datos:")
    info = get_data_info(df)
    for key, value in info.items():
        print(f"{key}: {value}")
