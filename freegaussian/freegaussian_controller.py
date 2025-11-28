from dataclasses import dataclass, field
from typing import Type, Tuple, Dict

import torch
import torch.nn as nn
from nerfstudio.configs import base_config as cfg
from nerfstudio.viewer.viewer_elements import ViewerVec3


@dataclass
class FreeGaussianControllerConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: FreeGaussianController)


class FreeGaussianController(nn.Module):
    def __init__(self, config: FreeGaussianControllerConfig):
        super().__init__()
        self.config = config

    def register_vector3(self, num_attributes: int):
        def register_individual_vector(id):
            def gui_cb(element: ViewerVec3):
                print(element.value)

            setattr(
                self,
                f"vector_{id}",
                ViewerVec3(name=f"atrb_{id}", default_value=(0.0, 0.0, 0.0), step=0.01, cb_hook=gui_cb),
            )

        self.num_attributes = num_attributes
        for i in range(num_attributes):
            register_individual_vector(i)

    def get_atrb_vals(self):
        attrb_vals = []
        for i in range(self.num_attributes):
            attrb_vals.append(torch.tensor(getattr(self, f"vector_{i}").value))
        return torch.stack(attrb_vals) * 0.1
