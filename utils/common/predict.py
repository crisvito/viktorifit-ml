import pandas as pd
# from .encode_utils import encode_value

def predict_user_input(user_input, models, scaler=None, scaler_features=None):
    """
    Predict user input menggunakan model-model yang diberikan.

    Parameters
    ----------
    user_input : dict
        Input user, key = nama kolom, value = nilai.
    models : dict
        Dictionary model {col_name: trained_model}.
    scaler : sklearn scaler, optional
        Scaler untuk fitur numerik. Jika None, tidak melakukan scaling.
    scaler_features : list, optional
        Nama-nama kolom yang akan di-scale.

    Returns
    -------
    dict
        Prediksi untuk setiap model/kolom.
    """
    # Encode string input
    input_values = {}
    for col, val in user_input.items():
        if isinstance(val, str) and col.endswith('_encoded'):
            val = encode_value(col, val)
        input_values[col] = val

    df_input = pd.DataFrame([input_values])

    # Scaling
    if scaler is not None and scaler_features is not None:
        df_input[scaler_features] = scaler.transform(df_input[scaler_features])

    # Predict
    predictions = {}
    for col, model in models.items():
        predictions[col] = model.predict(df_input)[0]

    return predictions
