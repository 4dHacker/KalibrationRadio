import math 
from math import comb

def polar1 (a):
    re = a.real # real Teil
    im = a.imag # imaginär Teil
    r = math.sqrt(re**2+im**2)
    phi = math.atan(im/re) # im Bogenmaß
    return (r,phi)

def polar2 (c):
    i = complex (0,1)
    r= np.abs (c)
    phi = -i *(np.log(c/r))
    return (r,phi)
