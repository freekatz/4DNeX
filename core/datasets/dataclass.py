import numpy as np
from dataclasses import dataclass


@dataclass
class Pointmap:
    xyz: np.ndarray = None        # [F, H, W, 3] — 3D coordinates
    rgb: np.ndarray = None        # [F, H, W, 3] — RGB values [0, 1]
    depth: np.ndarray = None      # [F, H, W] — depth maps (optional, from optimization)
    cams2world: np.ndarray = None # [F, 4, 4] — camera-to-world transforms (optional)
    K: np.ndarray = None          # [F, 3, 3] — intrinsic matrices (optional)

    @property
    def num_frames(self):
        return self.xyz.shape[0]

    @property
    def height(self):
        return self.xyz.shape[1]

    @property
    def width(self):
        return self.xyz.shape[2]
