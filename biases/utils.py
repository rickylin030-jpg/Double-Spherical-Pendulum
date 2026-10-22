from torch import Tensor
from scipy.spatial.transform import Rotation
import torch
import numpy as np
# from oil.utils.utils.import export
import random
import torch


class FixedSeedAll(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)
        self.pt_rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)
        torch.random.set_rng_state(self.pt_rng_state)

def rel_err(x: Tensor, y: Tensor) -> Tensor:
    return (((x - y) ** 2).sum() / ((x + y) ** 2).sum()).sqrt()
