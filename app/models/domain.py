from pydantic import BaseModel
from pathlib import Path
from typing import Literal

class MediaRef(BaseModel):
    namespace: Literal["inputs", "outputs"]
    path: str

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix
