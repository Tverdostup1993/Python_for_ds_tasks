import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_data(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Split the dataset into train and validation sets.

    Returns:
        Dict containing 'train_inputs', 'train_targets', 'val_inputs', 'val_targets'
    """
    X = df.drop(columns=[target_col, 'Surname'], errors='ignore')
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return {
        'train_inputs': X_train.reset_index(drop=True),
        'train_targets': y_train.reset_index(drop=True),
        'val_inputs': X_val.reset_index(drop=True),
        'val_targets': y_val.reset_index(drop=True)
    }


def impute_numeric(data: Dict[str, Any], numeric_cols: List[str]) -> SimpleImputer:
    """
    Fill missing values in numeric columns using the mean.

    Returns:
        Fitted imputer.
    """
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(data['train_inputs'][numeric_cols])

    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(data[f'{split}_inputs'][numeric_cols])

    return imputer


def scale_numeric(data: Dict[str, Any], numeric_cols: List[str], apply_scaling: bool) -> Optional[StandardScaler]:
    """
    Scale numeric columns using StandardScaler if apply_scaling is True.

    Returns:
        Fitted scaler or None.
    """
    if not apply_scaling:
        return None

    scaler = StandardScaler()
    scaler.fit(data['train_inputs'][numeric_cols])

    for split in ['train', 'val']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])

    return scaler


def encode_categorical(data: Dict[str, Any], categorical_cols: List[str]) -> Tuple[OneHotEncoder, List[str]]:
    """
    Apply one-hot encoding to categorical columns.

    Returns:
        Fitted encoder and list of final column names.
    """
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(data['train_inputs'][categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)
        data[f'{split}_inputs'] = pd.concat([
            data[f'{split}_inputs'].drop(columns=categorical_cols),
            encoded_df
        ], axis=1)

    return encoder, data['train_inputs'].columns.tolist()


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[StandardScaler], OneHotEncoder]:
    """
    Full preprocessing pipeline.

    Returns:
        X_train, y_train, X_val, y_val, input_cols, scaler, encoder
    """
    target_col = 'Exited'
    data = split_data(raw_df, target_col)

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()

    imputer = impute_numeric(data, numeric_cols)
    scaler = scale_numeric(data, numeric_cols, scaler_numeric)
    encoder, input_cols = encode_categorical(data, categorical_cols)

    return (
        data['train_inputs'],
        data['train_targets'],
        data['val_inputs'],
        data['val_targets'],
        input_cols,
        scaler,
        encoder
    )


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Process new data using trained scaler & encoder.

    Returns:
        Processed new DataFrame
    """
    df = new_df.drop(columns=['Surname'], errors='ignore')
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    df = df.reindex(columns=input_cols, fill_value=0)
    return df