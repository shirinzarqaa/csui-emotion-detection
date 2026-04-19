import json
import pandas as pd
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from the JSON file into a Pandas DataFrame.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If the JSON is a list of dicts:
    df = pd.DataFrame(data)
    
    # Target label: fallback to label_finegrained if new_label_fine_grained is missing
    if 'new_label_fine_grained' in df.columns:
        df['target'] = df['new_label_fine_grained']
    else:
        df['target'] = df['label_finegrained']
        
    # Drop rows without text or target
    df = df.dropna(subset=['text', 'target'])
    return df

def get_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns train, val, and test dataframes based on the 'splitting' column.
    """
    if 'splitting' not in df.columns:
        # Fallback to 80/10/10 split if splitting column isn't provided
        from sklearn.model_selection import train_test_split
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        return train, val, test

    train = df[df['splitting'].str.lower() == 'train']
    val = df[df['splitting'].str.lower() == 'validation']
    test = df[df['splitting'].str.lower() == 'test']
    
    # If validation is empty usually it might be called val or dev
    if val.empty:
        val = df[df['splitting'].str.lower().isin(['val', 'dev'])]
        
    return train, val, test

def prepare_data(filepath: str):
    df = load_data(filepath)
    train_df, val_df, test_df = get_splits(df)
    
    # Get all unique classes sorted
    classes = sorted(df['target'].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    id_to_class = {i: c for i, c in enumerate(classes)}
    
    # Build Fine to Basic mapping dictionary
    fine_to_basic = {}
    if 'new_label_basic' in df.columns and 'new_label_fine_grained' in df.columns:
        mapping_df = df[['new_label_fine_grained', 'new_label_basic']].drop_duplicates()
        fine_to_basic = dict(zip(mapping_df['new_label_fine_grained'], mapping_df['new_label_basic']))
    elif 'label_basic' in df.columns and 'label_finegrained' in df.columns:
        mapping_df = df[['label_finegrained', 'label_basic']].drop_duplicates()
        fine_to_basic = dict(zip(mapping_df['label_finegrained'], mapping_df['label_basic']))
    
    train_df['label_id'] = train_df['target'].map(class_to_id)
    val_df['label_id'] = val_df['target'].map(class_to_id)
    test_df['label_id'] = test_df['target'].map(class_to_id)
    
    return train_df, val_df, test_df, class_to_id, id_to_class, fine_to_basic
