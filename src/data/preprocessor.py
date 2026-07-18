import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config["preprocessing"]
        self.scaler = None
        self.selector = None

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        for c in cat_cols:
            df[c] = df[c].fillna("missing").astype("category").cat.codes
        return df

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if not self.config["normalize"]:
            return df
        method = self.config["normalization_method"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if fit:
            if method == "zscore":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if not self.config["feature_selection"]["enabled"]:
            return X
        method = self.config["feature_selection"]["method"]
        k = self.config["feature_selection"]["k_best"]
        if method == "mutual_info":
            self.selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
            X_selected = self.selector.fit_transform(X, y)
            mask = self.selector.get_support()
            return X.loc[:, mask]
        return X

    def balance(self, X: pd.DataFrame, y: pd.Series):
        if not self.config["handle_imbalance"]:
            return X, y
        strategy = self.config["sampling_strategy"]
        if strategy == "smote":
            smote = SMOTE(random_state=42)
            return smote.fit_resample(X, y)
        return X, y

    def extract_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        if "Flow ID" not in df.columns or "Timestamp" not in df.columns:
            print("[PREPROC] 'Flow ID' or 'Timestamp' not present — using existing CICIDS features instead")
            if "Packet Length" not in df.columns:
                if "Total Length of Fwd Packets" in df.columns and "Total Length of Bwd Packets" in df.columns:
                    pl = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
                else:
                    pl = pd.Series(0, index=df.index)
            else:
                pl = df["Packet Length"]
            out["packet_rate"] = df.get("Flow Packets/s", pd.Series(0, index=df.index))
            out["byte_rate"] = df.get("Flow Bytes/s", pd.Series(0, index=df.index))
            out["mean_packet_size"] = pl.groupby(df["Destination Port"]).transform("mean").fillna(pl.median())
            return pd.concat([df, out], axis=1)
        out["packet_rate"] = df.groupby("Flow ID")["Timestamp"].transform("count")
        out["byte_rate"] = df.groupby("Flow ID")["Packet Length"].transform("sum")
        out["mean_packet_size"] = df.groupby("Flow ID")["Packet Length"].transform("mean")
        return pd.concat([df, out], axis=1)
