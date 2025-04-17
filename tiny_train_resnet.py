import numpy as np
from PIL import Image
from resnet import ResNet

from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from tiny_utils import train, evaluate, fetch_mnist

