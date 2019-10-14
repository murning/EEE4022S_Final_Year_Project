import gccphat
from scipy import signal
import numpy as np

a = signal.unit_impulse(8000)
b = signal.unit_impulse(8000)
b = np.roll(b, 3)

def test_tdoa():   
    assert gccphat.tdoa(a,b,8000) == 0.000375

