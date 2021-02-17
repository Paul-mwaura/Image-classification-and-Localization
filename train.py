import torch
import random
import pandas as pd
import numpy as np
import torch.nn as nn
from efficientnet.model import EfficientNet

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


model = EfficientNet.from_pretrained(model_name)
model._fc = nn.Linear(model._fc.in_features, dls.c)
