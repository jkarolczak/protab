import os
import random

import numpy as np
import torch


def set_seed(seed: int | None = None) -> int:
    seed = seed or int(os.environ.get("PROTAB_RANDOM_SEED", 42))
    os.environ["PROTAB_RANDOM_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
