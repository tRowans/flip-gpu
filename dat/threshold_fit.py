import numpy as np
from scipy.optimize import curve_fit

def ansatz(p,a0,a1,a2,pth,v):
    x = (p-pth)L**(1/v)
    return a0 + a1*x + a2*(x**2)




