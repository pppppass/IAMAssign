import numpy
import scipy.sparse


def ana_sol(size, time):
    """Analytical solution"""
    n, t = size, time
    x, y = (numpy.indices((n-1, n-1)) + 1) / n
    u = (
          numpy.exp(-2.0 * numpy.pi**2 * t)
        * numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)
    )
    return u


def mat_lap(size):
    """Matrix of Laplacian"""
    n = size
    list_ = []
    for i in range(n-1):
        for j in range(n-1):
            list_.append((4.0, i, j, i, j))
            list_.append((-1.0, i, j, i, j-1))
            list_.append((-1.0, i, j, i, j+1))
            list_.append((-1.0, i, j, i-1, j))
            list_.append((-1.0, i, j, i+1, j))
    data = []
    rows = []
    cols = []
    for v, x1, x2, y1, y2 in list_:
        if 0 <= y1 < n-1 and 0 <= y2 < n-1:
            data.append(v)
            rows.append(x1*(n-1) + x2)
            cols.append(y1*(n-1) + y2)
    lap = scipy.sparse.coo_matrix(
        (data, (rows, cols)), shape=((n-1)**2, (n-1)**2))
    lap = lap.tocsr()
    return lap


def mat_int(size):
    """Matrix of interpolation operator"""
    n = size
    list_ = []
    for i in range(n):
        for j in range(n):
            list_.append((1.0 / 4.0, 2*i, 2*j, i, j))
            list_.append((1.0 / 4.0, 2*i, 2*j, i, j-1))
            list_.append((1.0 / 4.0, 2*i, 2*j, i-1, j))
            list_.append((1.0 / 4.0, 2*i, 2*j, i-1, j-1))
            list_.append((1.0 / 2.0, 2*i, 2*j+1, i, j))
            list_.append((1.0 / 2.0, 2*i, 2*j+1, i-1, j))
            list_.append((1.0 / 2.0, 2*i+1, 2*j, i, j))
            list_.append((1.0 / 2.0, 2*i+1, 2*j, i, j-1))
            list_.append((1.0, 2*i+1, 2*j+1, i, j))
    data = []
    rows = []
    cols = []
    for v, x1, x2, y1, y2 in list_:
        if x1 < 2*n-1 and x2 < 2*n-1:
            if 0 <= y1 < n-1 and 0 <= y2 < n-1:
                data.append(v)
                rows.append(x1*(2*n-1) + x2)
                cols.append(y1*(n-1) + y2)
    int_ = scipy.sparse.coo_matrix(
        (data, (rows, cols)), shape=((2*n-1)**2, (n-1)**2))
    int_ = int_.tocsr()
    return int_


def quad_simp(step, func):
    """Simpson quadrature"""
    h, u = step, func
    i = (
        16.0 * numpy.sum(func[::2, ::2])
        + 8.0 * (numpy.sum(func[::2, 1::2]) + numpy.sum(func[1::2, ::2]))
        + 4.0 * numpy.sum(func[1::2, 1::2])
    ) * step**2 / 9.0
    return i


def int_bil(size, sub, func):
    """Bilinear interpolation"""
    n, k, u = size, sub, func
    c = numpy.zeros((n, k, n, k))
    x, y = numpy.indices((k, k))
    p = u[:, None, :, None]
    l = (x * y / k**2)[None, :, None, :]
    c[:-1, :, :-1, :] += p * l
    l = (x * (k-y) / k**2)[None, :, None, :]
    c[:-1, :, 1:, :] += p * l
    l = ((k-x) * y / k**2)[None, :, None, :]
    c[1:, :, :-1, :] += p * l
    l = ((k-x) * (k-y) / k**2)[None, :, None, :]
    c[1:, :, 1:, :] += p * l
    c = c.reshape((n*k, n*k))[1:, 1:]
    return c


def calc_err(size, sub, func, time):
    """Calculate errors"""
    n, k, u_h = size, sub, func
    h = 1.0 / (size * sub)
    sol_ana = ana_sol(size*sub, time)
    sol_test = int_bil(n, k, u_h)
    len_ana = numpy.sqrt(quad_simp(h, sol_ana**2))
    err_ana = numpy.sqrt(quad_simp(h, (sol_ana - sol_test)**2))
    rel_ana = err_ana / len_ana
    return (err_ana, rel_ana)


def calc_err_std(size, sub, func, time, std):
    """Calculate errors with respect to a given standard"""
    n, k, u_h, u_s = size, sub, func, std
    h = 1.0 / (size * sub)
    sol_ana = ana_sol(size*sub, time)
    sol_test = int_bil(n, k, u_h)
    sol_std = int_bil(n, k, u_s)
    len_ana = numpy.sqrt(quad_simp(h, sol_ana**2))
    err_ana = numpy.sqrt(quad_simp(h, (sol_ana - sol_test)**2))
    rel_ana = err_ana / len_ana
    len_std = numpy.sqrt(quad_simp(h, sol_std**2))
    err_std = numpy.sqrt(quad_simp(h, (sol_std - sol_test)**2))
    rel_std = err_std / len_std
    return (err_ana, rel_ana, err_std, rel_std)
