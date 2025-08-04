from pathlib import Path
from threading import Lock
from typing import Optional

import pandas as pd


class DataBuffer:
    def __init__(self, buffer_path: Path, buffer_size: int):
        self.buffer_path = buffer_path
        self.buffer_size = buffer_size
        self.buffer = None

        if self.buffer_path.exists():
            self.buffer = pd.read_csv(self.buffer_path)

    def append(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.buffer is None:
            self.buffer = data
        else:
            self.buffer = pd.concat([self.buffer, data], ignore_index=True)
            self.buffer = self.buffer[-self.buffer_size :]

        self.buffer.to_csv(self.buffer_path, index=False)

        if len(self.buffer) >= self.buffer_size:
            return self.buffer.copy()

        return None

    def clear(self):
        self.buffer = None
        if self.buffer_path.exists():
            self.buffer_path.unlink()

    def get_data(self) -> pd.DataFrame:
        return self.buffer.copy() if self.buffer is not None else pd.DataFrame()
