from pathlib import Path
from threading import Lock

import pandas as pd

from src.monitoring.utils import DataBuffer


class DriftState:
    def __init__(self, buffer_path, buffer_size: int):
        buffer_path = Path(buffer_path)
        buffer_path.mkdir(parents=True, exist_ok=True)

        self.input_buffer = DataBuffer(buffer_path / "input_buffer.csv", buffer_size)
        self.pred_buffer = DataBuffer(buffer_path / "pred_buffer.csv", buffer_size)
        self.lock = Lock()

    def add_input(self, input_data: pd.DataFrame, proba: pd.DataFrame) -> bool:
        """
        Adds new data and predictions to buffers. Returns True if buffers are full.
        """
        with self.lock:
            self.input_buffer.append(input_data)
            pred_df = proba.to_frame(name="proba")

            full_buffer = self.pred_buffer.append(pred_df)
            return full_buffer is not None

    def reset(self):
        with self.lock:
            self.input_buffer.clear()
            self.pred_buffer.clear()

    def get_buffer_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with self.lock:
            return self.input_buffer.get_data(), self.pred_buffer.get_data()
