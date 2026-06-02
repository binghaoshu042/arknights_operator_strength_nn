import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_profession_data(file_path="data/raw/operators.csv", profession=None, test_size=0.2, random_state=42):
    """
    Loads raw operator data and filters it by a specific profession (e.g. Guard, Caster).
    Splits the filtered data into training and testing sets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please run generate_mock_data.py first.")
        
    df = pd.read_csv(file_path)
    
    if profession:
        df_filtered = df[df["profession"].str.lower() == profession.lower()].copy()
        if len(df_filtered) == 0:
            raise ValueError(f"No operators found for profession: {profession}")
    else:
        df_filtered = df.copy()
        
    # Features and target split
    # Inputs: atk, def, res, dps, dph, type, branch
    # Target: cup
    feature_cols = [
        "profession", "atk", "def", "res", "dps", "dph", "type", "branch",
        "control_coverage", "buff_amp", "total_damage", "sp_cost", "init_sp"
    ]
    target_col = "cup"
    
    X = df_filtered[feature_cols].copy()
    y = df_filtered[target_col].copy()
    
    # Train-test split
    # Stratify only if every class has at least 2 samples, otherwise train_test_split will fail.
    class_counts = y.value_counts()
    stratify_target = y if (class_counts.min() >= 2 and len(class_counts) > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_target
    )
    
    return X_train, X_test, y_train, y_test
