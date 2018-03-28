import numpy

def calc_newton(x, y):
    n = x.size
    c = numpy.zeros(n)
    
    c[0] = y[0]
    for i in range(1, n):
        u, v = 0., 1.
        for j in range(i-1, -1, -1):
            u = u * (x[i] - x[j]) + c[j]
            v = v * (x[i] - x[j])
        c[i] = (y[i] - u) / v
    
    return c

def calc_value(x, c, u):
    n = x.size
    m = u.size
    v = numpy.zeros(m)
    
    for i in range(n-1, -1, -1):
        v = v * (u - x[i]) + c[i]
    
    return v