import pandas as pd
import numpy as np
from pathlib import Path
import yaml


class DataCollector:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.raw_path = Path(self.config["data"]["raw_path"])

    def load_cicids2017(self) -> pd.DataFrame:
        path = Path(self.config["data"]["datasets"]["cicids2017"]["path"])
        files = self.config["data"]["datasets"]["cicids2017"]["files"]
        frames = []
        for f in files:
            filepath = path / f
            if filepath.exists():
                df = pd.read_csv(filepath)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_unsw_nb15(self) -> pd.DataFrame:
        path = Path(self.config["data"]["datasets"]["unsw_nb15"]["path"])
        files = [
            "UNSW_NB15_training-set.csv",
            "UNSW_NB15_testing-set.csv",
        ]
        frames = []
        for f in files:
            filepath = path / f
            if filepath.exists():
                df = pd.read_csv(filepath)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def save_processed(self, df: pd.DataFrame, filename: str):
        out = Path(self.config["data"]["processed_path"])
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / f"{filename}.parquet", index=False)
