import numpy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import utils
import exts


def pde_exp(size_h, size_t, init, time):
    """Evolve the PDE with explicit scheme and Python SuperLU binding"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye - dt / dx**2 * lap
    for i in range(m):
        u = a.dot(u)
    u = u.reshape((n-1, n-1))
    return u


def pde_imp(size_h, size_t, init, time):
    """Evolve the PDE with implicit scheme and Python SuperLU binding"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    a_fac = scipy.sparse.linalg.splu(a.transpose())
    for i in range(m):
        u = a_fac.solve(u)
    u = u.reshape((n-1, n-1))
    return u


def pde_cn(size_h, size_t, init, time):
    """Evolve the PDE with Crank-Nicolson scheme and Python SuperLU binding"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye - dt / dx**2 / 2.0 * lap
    b = eye + dt / dx**2 / 2.0 * lap
    b_fac = scipy.sparse.linalg.splu(b.transpose())
    for i in range(m):
        u = a.dot(u)
        u = b_fac.solve(u)
    u = u.reshape((n-1, n-1))
    return u


def pde_imp_cho_py(size_h, size_t, init, time):
    """Evolve the PDE using implicit scheme, Cholesky decomposition and Python"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    a = a.todia()
    ab = numpy.zeros((n, (n-1)**2))
    ab[0] = a.diagonal(k=0)
    ab[1, :-1] = a.diagonal(k=1)
    ab[n-1, :1-n] = a.diagonal(k=n-1)
    l = scipy.linalg.cholesky_banded(ab, lower=True)
    l_t = numpy.zeros((n, (n-1)**2))
    for i in range(n):
        l_t[n-i-1, i:] = l[i, :(n-1)**2-i]
    ctr = 0
    for i in range(m):
        u = scipy.linalg.solve_banded((n-1, 0), l, u)
        u = scipy.linalg.solve_banded((0, n-1), l_t, u)
        ctr += 1
    u = u.reshape((n-1, n-1))
    return u, ctr


def solve_gs(mat, mat_u, mat_dl_fac, vec, init, eps, chk):
    """Gauss-Seidel to solve a system"""
    b, x = vec, init
    a, a_u, a_dl_fac = mat, mat_u, mat_dl_fac
    w = a_dl_fac.solve(b)
    delta_0 = numpy.linalg.norm(b - a.dot(x), 2)
    ctr = 0
    while True:
        x = a_dl_fac.solve(a_u.dot(x)) + w
        ctr += 1
        if ctr % chk == 0:
            delta = numpy.linalg.norm(b - a.dot(x), 2)
            if delta < delta_0 * eps:
                break
    return x, ctr


def pde_imp_gs_py(size_h, size_t, init, time, eps, chk):
    """Evolve the PDE using implicit scheme, Gauss-Seidel and Python"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    a_u = -scipy.sparse.triu(a, k=1, format="csr")
    a_dl = scipy.sparse.tril(a.transpose(), k=0, format="csc")
    a_dl_fac = scipy.sparse.linalg.splu(a_dl, permc_spec="NATURAL")
    ctr = 0
    for i in range(m):
        u, temp = solve_gs(a, a_u, a_dl_fac, u, u, eps, chk)
        ctr += temp
    u = u.reshape((n-1, n-1))
    return u, ctr


def solve_sd(mat, vec, init, eps):
    """Steepest descent to solve a system"""
    a, b, x = mat, vec, init
    r = b - a.dot(x)
    rho = r.dot(r)
    rho_0 = rho
    ctr = 0
    while True:
        w = a.dot(r)
        alpha = rho / r.dot(w)
        x = x + alpha * r
        r = r - alpha * w
        rho = r.dot(r)
        ctr += 1
        if rho < rho_0 * eps**2:
            break
    return x, ctr


def pde_imp_sd_py(size_h, size_t, init, time, eps):
    """Evolve the PDE using implicit scheme, steepest descent and Python"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    ctr = 0
    for i in range(m):
        u, temp = solve_sd(a, u, u, eps)
        ctr += temp
    u = u.reshape((n-1, n-1))
    return u, ctr


def solve_cg(mat, vec, init, eps):
    """Conjugate gradient to solve a system"""
    a, b, x = mat, vec, init
    r = b - a.dot(x)
    rho = r.dot(r)
    rho_0 = rho
    ctr = 0
    while True:
        if ctr == 0:
            p = r
        else:
            beta = rho / rho_old
            p = r + beta * p
        w = a.dot(p)
        alpha = rho / p.dot(w)
        x = x + alpha * p
        r = r - alpha * w
        rho_old = rho
        rho = r.dot(r)
        ctr += 1
        if rho < rho_0 * eps**2:
            break
    return x, ctr


def pde_imp_cg_py(size_h, size_t, init, time, eps):
    """Evolve the PDE using implicit scheme, conjugate gradient and Python"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    ctr = 0
    for i in range(m):
        u, temp = solve_cg(a, u, u, eps)
        ctr += temp
    u = u.reshape((n-1, n-1))
    return u, ctr


def step_mg(mats, mat_us, mat_dl_facs, ints, nums, vec, init):
    """Multigrid step"""
    b, x = vec, init
    a, a_u, a_dl_fac = mats[0], mat_us[0], mat_dl_facs[0]
    w = a_dl_fac.solve(b)
    for j in range(nums[0]):
        x = a_dl_fac.solve(a_u.dot(x)) + w
    if len(ints) > 0:
        i = ints[0]
        r = b - a.dot(x)
        b_new = i.transpose().dot(r)
        x_new = numpy.zeros(i.shape[1])
        x_new = step_mg(
            mats[1:], mat_us[1:], mat_dl_facs[1:],
            ints[1:], nums[1:],
            b_new, x_new
        )
        x += i.dot(x_new)
    return x


def solve_mg(mats, mat_us, mat_dl_facs, ints, nums, vec, init, eps, chk):
    """Multigrid to solve a system"""
    b, x = vec, init
    a = mats[0]
    r = b - a.dot(x)
    delta_0 = numpy.linalg.norm(r, 2)
    ctr = 0
    while True:
        x = step_mg(mats, mat_us, mat_dl_facs, ints, nums, b, x)
        ctr += 1
        if ctr % chk == 0:
            r = b - a.dot(x)
            delta = numpy.linalg.norm(r, 2)
            if delta < delta_0 * eps:
                break
    return x, ctr


def pde_imp_mg_py(size_h, size_t, init, time, nums, eps, chk):
    """Evolve the PDE using implicit scheme, multigrid and Python"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    mats, mat_us, mat_dl_facs, ints = [], [], [], []
    temp, j = n, 0
    while True:
        a_u = -scipy.sparse.triu(a, k=1, format="csr")
        a_dl = scipy.sparse.tril(a.transpose(), k=0, format="csc")
        a_dl_fac = scipy.sparse.linalg.splu(a_dl, permc_spec="NATURAL")
        mats.append(a)
        mat_us.append(a_u)
        mat_dl_facs.append(a_dl_fac)
        j += 1
        if j >= len(nums):
            break
        temp //= 2
        i = utils.mat_int(temp)
        ints.append(i)
        a = i.transpose().dot(a.dot(i))
    ctr = 0
    for i in range(m):
        u, temp = solve_mg(mats, mat_us, mat_dl_facs, ints, nums, u, u, eps, chk)
        ctr += temp
    u = u.reshape((n-1, n-1))
    return u, ctr


def pde_imp_cho_c(size_h, size_t, init, time):
    """Evolve the PDE using implicit scheme, Cholesky decomposition and C"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    ctr = exts.pde_imp_cho_wrapper(a.data, a.indices, a.indptr, u, n-1, m)
    u = u.reshape((n-1, n-1))
    return u, ctr


def pde_imp_gs_c(size_h, size_t, init, time, eps, chk):
    """Evolve the PDE using implicit scheme, Gauss-Seidel and C"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    ctr = exts.pde_imp_gs_wrapper(a.data, a.indices, a.indptr, u, eps, chk, m)
    u = u.reshape((n-1, n-1))
    return u, ctr


def pde_imp_sd_c(size_h, size_t, init, time, eps):
    """Evolve the PDE using implicit scheme, steepest descent and C"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    ctr = exts.pde_imp_sd_wrapper(a.data, a.indices, a.indptr, u, eps, m)
    u = u.reshape((n-1, n-1))
    return u, ctr


def pde_imp_cg_c(size_h, size_t, init, time, eps):
    """Evolve the PDE using implicit scheme, conjugate gradient and C"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    ctr = exts.pde_imp_cg_wrapper(a.data, a.indices, a.indptr, u, eps, m)
    u = u.reshape((n-1, n-1))
    return u, ctr


def pde_imp_mg_c(size_h, size_t, init, time, num, eps, chk):
    """Evolve the PDE using implicit scheme, multigrid and C"""
    n, m, u, t = size_h, size_t, init, time
    u = u.flatten()
    num = numpy.array(num, dtype=numpy.int32)
    dx, dt = 1.0 / n, t / m
    lap = utils.mat_lap(n)
    eye = scipy.sparse.eye((n-1)**2).tocsr()
    a = eye + dt / dx**2 * lap
    datas, inds, ptrs = [], [], []
    datas.append(a.data)
    inds.append(a.indices)
    ptrs.append(a.indptr)
    temp = n
    for j in range(len(num) - 1):
        temp //= 2
        i = utils.mat_int(temp)
        datas.append(i.data)
        inds.append(i.indices)
        ptrs.append(i.indptr)
    ctr = exts.pde_imp_mg_wrapper(n, datas, inds, ptrs, u, num, eps, chk, m)
    u = u.reshape((n-1, n-1))
    return u, ctr
