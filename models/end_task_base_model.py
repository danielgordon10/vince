import abc
from typing import Optional, Dict, Tuple

import numpy as np
import torch

from models.base_model import BaseModel


class EndTaskBaseModel(BaseModel, abc.ABC):
    def loss(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[Tuple[float, torch.Tensor]]]:
        raise NotImplementedError

    def get_metrics(self, network_outputs: Optional[Dict]) -> Dict[str, Optional[float]]:
        raise NotImplementedError

    def get_image_output(self, network_outputs) -> Dict[str, np.ndarray]:
        raise NotImplementedError
