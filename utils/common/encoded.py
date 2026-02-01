import os
import pandas as pd

def encode_value(val, col_name=None, encoder_path=None, df_encoder=None):
    """
    Mengubah string label menjadi angka encoded.
    
    Parameters
    ----------
    val : str
        Nilai label yang akan di-encode
    encoder_path : str, optional
        Path ke file CSV encoder, misal 'data/data1/encoder/Exercises_encoder.csv'
    col_name : str, optional
        Nama kolom target, misal 'Exercises_encoded' (untuk membangun path otomatis jika encoder_path None)
    df_encoder : pd.DataFrame, optional
        DataFrame mapping label -> encoded. Kalau dikirim, file CSV tidak dibaca.
    
    Returns
    -------
    int/float
        Nilai encoded, nan jika tidak ditemukan
    """
    if df_encoder is not None:
        mapping = dict(zip(df_encoder['label'], df_encoder['encoded']))
    else:
        if encoder_path is None:
            if col_name is None:
                raise ValueError("Salah satu dari 'encoder_path' atau 'col_name' harus diisi.")
            encoder_path = os.path.join("data", "data1", "encoder", f"{col_name.replace('_encoded','')}_encoder.csv")
        df_encoder = pd.read_csv(encoder_path)
        mapping = dict(zip(df_encoder['label'], df_encoder['encoded']))
    
    return mapping.get(val, float('nan'))


def decode_value(val, encoder_path=None, col_name=None, df_encoder=None):
    """
    Mengubah angka encoded menjadi string label asli.
    
    Parameters
    ----------
    val : int/float
        Nilai encoded yang akan dikembalikan ke string
    encoder_path : str, optional
        Path ke file CSV encoder
    col_name : str, optional
        Nama kolom target, misal 'Exercises_encoded'
    df_encoder : pd.DataFrame, optional
        DataFrame mapping encoded -> label. Kalau dikirim, file CSV tidak dibaca.
    
    Returns
    -------
    str / None
        Nilai label asli, None jika tidak ditemukan
    """
    if df_encoder is not None:
        mapping = dict(zip(df_encoder['encoded'], df_encoder['label']))
    else:
        if encoder_path is None:
            if col_name is None:
                raise ValueError("Salah satu dari 'encoder_path' atau 'col_name' harus diisi.")
            encoder_path = os.path.join("data", "data1", "encoder", f"{col_name.replace('_encoded','')}_encoder.csv")
        df_encoder = pd.read_csv(encoder_path)
        mapping = dict(zip(df_encoder['encoded'], df_encoder['label']))
    
    return mapping.get(val, None)
