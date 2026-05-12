import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from src.utils.preprocessing import preprocess_for_traditional, preprocess_for_deep_learning, preprocess_for_transformers

FINE_TO_BASIC_TAXONOMY = {
    'acceptance': 'joy',
    'admiration': 'joy',
    'aggressiveness': 'anger',
    'amusement': 'joy',
    'annoyance': 'anger',
    'anticipation': 'fear',
    'approval': 'joy',
    'attraction': 'love',
    'broken-heart': 'sadness',
    'caring': 'love',
    'compassion': 'sadness',
    'confusion': 'surprise',
    'contentment': 'joy',
    'disapproval': 'anger',
    'disgust': 'anger',
    'distrust': 'fear',
    'embarrassment': 'sadness',
    'fear': 'fear',
    'fears-confirmed': 'fear',
    'feeling moved': 'sadness',
    'gratitude': 'joy',
    'grief': 'sadness',
    'happy-for': 'joy',
    'hopelessness': 'sadness',
    'joy': 'joy',
    'longing': 'love',
    'love': 'love',
    'lust': 'love',
    'nervousness': 'fear',
    'no emotion': 'no emotion',
    'nonsexual desire': 'love',
    'optimism': 'joy',
    'pensiveness': 'sadness',
    'pity': 'sadness',
    'pride': 'joy',
    'rage': 'anger',
    'realization': 'surprise',
    'relief': 'joy',
    'remorse': 'sadness',
    'restlessness': 'fear',
    'sincerity': 'joy',
    'submission': 'fear',
    'suffering': 'sadness',
    'surprise': 'surprise',
    'worry': 'fear',
}

BASIC_LABELS = sorted(set(FINE_TO_BASIC_TAXONOMY.values()))
FINE_LABELS = sorted(FINE_TO_BASIC_TAXONOMY.keys())

BASIC_TO_ID = {label: idx for idx, label in enumerate(BASIC_LABELS)}
ID_TO_BASIC = {idx: label for label, idx in BASIC_TO_ID.items()}
FINE_TO_ID = {label: idx for idx, label in enumerate(FINE_LABELS)}
ID_TO_FINE = {idx: label for label, idx in FINE_TO_ID.items()}
NUM_BASIC = len(BASIC_LABELS)
NUM_FINE = len(FINE_LABELS)


def parse_labels(label_str: str, label_set: set, normalize: bool = True) -> List[str]:
    if pd.isna(label_str) or not label_str:
        return []
    labels = [l.strip() for l in str(label_str).split(',')]
    if normalize:
        labels = [l.lower() for l in labels]
    return [l for l in labels if l in label_set]


def build_binary_label_matrix(labels_list: List[List[str]], label_to_id: Dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(labels_list), len(label_to_id)), dtype=np.int32)
    for i, labels in enumerate(labels_list):
        for label in labels:
            if label in label_to_id:
                matrix[i, label_to_id[label]] = 1
    return matrix


def load_data(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.dropna(subset=['text', 'new_label_basic', 'new_label_fine_grained'])
    return df


def get_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if 'splitting' not in df.columns:
        from sklearn.model_selection import train_test_split
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        return train, val, test

    train = df[df['splitting'].str.lower() == 'train']
    val = df[df['splitting'].str.lower() == 'validation']
    test = df[df['splitting'].str.lower() == 'test']

    if val.empty:
        val = df[df['splitting'].str.lower().isin(['val', 'dev'])]

    return train, val, test


def prepare_data(filepath: str, use_fine_grained: bool = True, preprocessing_mode: str = 'traditional'):
    df = load_data(filepath)
    
    # PREPROCESSING SEBELUM SPLIT (flow user) - mode berbeda per pipeline
    preprocess_fn = {
        'traditional': preprocess_for_traditional,
        'deep_learning': preprocess_for_deep_learning,
        'transformers': preprocess_for_transformers,
    }[preprocessing_mode]
    
    df['preprocessed_text'] = df['text'].apply(preprocess_fn)
    
    train_df, val_df, test_df = get_splits(df)

    # Parse multi-labels for each row
    train_basic_raw = [parse_labels(t, set(BASIC_LABELS)) for t in train_df['new_label_basic']]
    val_basic_raw = [parse_labels(t, set(BASIC_LABELS)) for t in val_df['new_label_basic']]
    test_basic_raw = [parse_labels(t, set(BASIC_LABELS)) for t in test_df['new_label_basic']]

    train_fine_raw = [parse_labels(t, set(FINE_LABELS)) for t in train_df['new_label_fine_grained']]
    val_fine_raw = [parse_labels(t, set(FINE_LABELS)) for t in val_df['new_label_fine_grained']]
    test_fine_raw = [parse_labels(t, set(FINE_LABELS)) for t in test_df['new_label_fine_grained']]

    # Build binary label matrices
    y_train_basic = build_binary_label_matrix(train_basic_raw, BASIC_TO_ID)
    y_val_basic = build_binary_label_matrix(val_basic_raw, BASIC_TO_ID)
    y_test_basic = build_binary_label_matrix(test_basic_raw, BASIC_TO_ID)

    y_train_fine = build_binary_label_matrix(train_fine_raw, FINE_TO_ID)
    y_val_fine = build_binary_label_matrix(val_fine_raw, FINE_TO_ID)
    y_test_fine = build_binary_label_matrix(test_fine_raw, FINE_TO_ID)

    return (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        BASIC_TO_ID, ID_TO_BASIC,
        FINE_TO_ID, ID_TO_FINE,
        FINE_TO_BASIC_TAXONOMY,
    )