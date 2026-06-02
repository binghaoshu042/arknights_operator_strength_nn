import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class OperatorFeatureExtractor:
    def __init__(self):
        # Expanded to 10 continuous numeric features
        self.num_cols = ["atk", "def", "res", "dps", "dph", "control_coverage", "buff_amp", "total_damage", "sp_cost", "init_sp"]
        self.cat_cols = ["profession", "type", "branch"]
        self.scaler = StandardScaler()
        # handle_unknown='ignore' prevents crashes when predicting unseen categories in Streamlit
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.target_mapping = {
            "小杯": 0,
            "中杯": 1,
            "大杯": 2,
            "超大杯": 3
        }
        self.reverse_target_mapping = {v: k for k, v in self.target_mapping.items()}
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, X_df):
        """Fits the numerical scaler and categorical encoder on the training dataframe X."""
        df = X_df.copy()
        
        # Apply the Arts damage rule during fit to ensure correct scaling baseline
        df.loc[df["type"] == "法术", "dph"] = 0.0
        
        # Fit numerical
        self.scaler.fit(df[self.num_cols])
        
        # Fit categorical
        self.encoder.fit(df[self.cat_cols])
        
        # Build feature name list for reference and transparency
        encoded_cat_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + encoded_cat_names
        
        self.is_fitted = True
        return self
        
    def transform(self, X_df):
        """Transforms input dataframe X into preprocessed feature array."""
        if not self.is_fitted:
            raise ValueError("Feature extractor is not fitted yet. Call fit() first.")
            
        df = X_df.copy()
        
        # KEY RULE: Mask DPH for Arts damage operators (set to 0.0)
        df.loc[df["type"] == "法术", "dph"] = 0.0
        
        # Scale numerical features
        num_feats = self.scaler.transform(df[self.num_cols])
        
        # Encode categorical features
        cat_feats = self.encoder.transform(df[self.cat_cols])
        
        # Concatenate features
        processed_feats = np.hstack((num_feats, cat_feats))
        
        return processed_feats
        
    def fit_transform(self, X_df):
        return self.fit(X_df).transform(X_df)
        
    def encode_labels(self, y_series):
        """Converts text labels ('超大杯', '大杯', etc.) into integer class indices (0, 1, 2, 3)."""
        return y_series.map(self.target_mapping).values
        
    def decode_labels(self, y_indices):
        """Converts integer class indices (0, 1, 2, 3) back into text labels ('超大杯', '大杯', etc.)."""
        if isinstance(y_indices, (int, np.integer)):
            return self.reverse_target_mapping[int(y_indices)]
        return [self.reverse_target_mapping[int(idx)] for idx in y_indices]
        
    def get_input_dim(self):
        """Returns the number of preprocessed features."""
        if not self.is_fitted:
            return 0
        return len(self.feature_names)
        
    def save(self, filepath):
        """Saves the fitted feature extractor to a pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath):
        """Loads a pre-saved feature extractor from a pickle file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
        with open(filepath, "rb") as f:
            extractor = pickle.load(f)
        return extractor
