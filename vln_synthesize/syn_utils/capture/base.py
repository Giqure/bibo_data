from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any

import numpy as np

class Sensor(ABC):
    world: Any = None
    stage: Any = None
    path_states: Any = None
    max_capture_paths: int = 0
    indices: np.ndarray | None = None
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    path_dir: str = ""
    meta: dict[str, Any] = {}
    
    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def collect(self):
        raise NotImplementedError

