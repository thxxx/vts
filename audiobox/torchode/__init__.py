"""A parallel ODE solver for PyTorch"""

__version__ = "1.0.0"

from torchode.interface import register_method
from torchode.single_step_methods.dopri5 import Dopri5
from torchode.single_step_methods.heun import Heun
from torchode.single_step_methods.midpoint import Midpoint
from torchode.single_step_methods.tsit5 import Tsit5

register_method("heun", Heun)
register_method("midpoint", Midpoint)
register_method("dopri5", Dopri5)
register_method("tsit5", Tsit5)
