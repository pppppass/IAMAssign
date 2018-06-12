import numpy


def newton(f, f_, x, eps, omega=1.0e6):
    ctr = 0
    success = False
    while True:
        if numpy.abs(f(x)) < eps:
            success = True
            break
        elif numpy.abs(x) > omega:
            break
        x -= f(x) / f_(x)
        ctr += 1
    return x, ctr, success


def ode_explicit(f, n, x):
    for i in range(n):
        t = numpy.float(i / n)
        x += f(x, t) / n
    return x


def ode_implicit_iterative(f, n, x, eps):
    h = numpy.float(1.0 / n)
    for i in range(n):
        t = numpy.float(i / n)
        x_new = x + f(x, t) / n
        while True:
            x_try = x + f(x_new, t + h) / n
            if (numpy.abs(x_new - x_try) < eps):
                break
            x_new = x_try
        x = x_new
    return x


def ode_trapezoid_iterative(f, n, x, eps):
    h = numpy.float(1.0 / n)
    for i in range(n):
        t = numpy.float(i / n)
        f_val = f(x, t)
        x_new = x + f_val / n
        while True:
            x_try = x + (f_val + f(x_new, t + h)) / n / 2.0
            if (numpy.abs(x_new - x_try) < eps):
                break
            x_new = x_try
        x = x_new
    return x


def ode_runge_kutta_4(f, n, x):
    h = numpy.float(1.0 / n)
    for i in range(n):
        t = numpy.float(i / n)
        k1 = f(x, t)
        k2 = f(x + k1 * h / 2.0, t + h / 2.0)
        k3 = f(x + k2 * h / 2.0, t + h / 2.0)
        k4 = f(x + k3 * h, t + h)
        x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * h / 6.0
    return x
