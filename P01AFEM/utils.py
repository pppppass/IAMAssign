import numpy

def quadrature_simpson(func, a, b, n):
    c = numpy.zeros(2*n+1)
    c[::2] = 2.0
    c[0] = 1.0
    c[2*n] = 1.0
    c[1::2] = 4.0
    c = c / numpy.sum(c) * (b - a)
    v = func(numpy.linspace(a, b, 2*n+1))
    i = numpy.dot(v, c)
    return i

def quadrature(func, a, b, eps=0.001):
    n = int((b - a) / eps) + 1
    i = quadrature_simpson(func, a, b, n)
    return i

def matrix_dirichlet(x):
    step = x[1:] - x[:-1]
    step_inv = 1.0 / step
    d = step_inv[:-1] + step_inv[1:]
    l = -step_inv[1:-1]
    u = -step_inv[1:-1]
    return d, l, u

def matrix_robin(x, k_start, k_end):
    step = x[1:] - x[:-1];
    step_inv = 1.0 / step
    d = numpy.hstack([0.0, step_inv]) + numpy.hstack([step_inv, 0.0])
    d[0] += k_start
    d[-1] -= k_end
    l = -step_inv
    u = -step_inv
    return d, l, u

def solve_chase(d, l, u, b):
    n = d.shape[0]
    for i in range(n-1):
        d[i+1] -= l[i] * u[i] / d[i]
        b[i+1] -= b[i] * l[i] / d[i]
    for i in range(n-1, 0, -1):
        b[i] /= d[i]
        b[i-1] -= b[i] * u[i-1]
    b[0] /= d[0]

def vector_dirichlet(func, x):
    n = x.shape[0] - 1
    b = numpy.zeros((n-1))
    for i in range(n-1):
        b[i] = (
              quadrature(lambda t: func(t) * (t - x[i]), x[i], x[i+1]) / (x[i+1] - x[i])
            + quadrature(lambda t: func(t) * (x[i+2] - t), x[i+1], x[i+2]) / (x[i+2] - x[i+1])
        )
    return b

def vector_robin(func, x):
    n = x.shape[0] - 1
    b = numpy.zeros((n+1))
    b[0] = quadrature(lambda t: func(t) * (x[1] - t), x[0], x[1]) / (x[1] - x[0])
    for i in range(n-1):
        b[i+1] = (
              quadrature(lambda t: func(t) * (t - x[i]), x[i], x[i+1]) / (x[i+1] - x[i])
            + quadrature(lambda t: func(t) * (x[i+2] - t), x[i+1], x[i+2]) / (x[i+2] - x[i+1])
        )
    b[-1] = quadrature(lambda t: func(t) * (x[-1] - t), x[-2], x[-1]) / (x[-1] - x[-2])
    return b

def error(func, x):
    n = x.shape[0] - 1
    eta = numpy.zeros((n))
    for i in range(n):
        eta[i] = numpy.sqrt(quadrature(lambda t: func(t)**2, x[i], x[i+1])) * (x[i+1] - x[i])
    return eta