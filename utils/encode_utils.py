import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
ENCODER_PATH = os.path.join(DATA_PATH, "encoder")

def encode_value(col_name, val):
    """
    Mengubah string label menjadi angka encoded.
    col_name : str
        Nama kolom target, misal 'Exercises_encoded'
    val : str
        Nilai label yang akan di-encode
    """
    file_path = os.path.join(ENCODER_PATH, f"{col_name.replace('_encoded','')}_encoder.csv")
    df = pd.read_csv(file_path)
    mapping = dict(zip(df['label'], df['encoded']))
    return mapping.get(val, float('nan'))

def decode_value(col_name, val):
    """
    Mengubah angka encoded menjadi string label asli.
    col_name : str
        Nama kolom target, misal 'Exercises_encoded'
    val : int/float
        Nilai encoded yang akan dikembalikan ke string
    """
    file_path = os.path.join(ENCODER_PATH, f"{col_name.replace('_encoded','')}_encoder.csv")
    df = pd.read_csv(file_path)
    mapping = dict(zip(df['encoded'], df['label']))
    return mapping.get(val, None)
