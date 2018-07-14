#include "Exts.h"

// Calculate the residue vector
void calc_res(int size, sparse_matrix_t mat, double* vec, double* sol, double* res)
{
    int n = size;
    double* b = vec, * x = sol;
    sparse_matrix_t a = mat;
    double* r = res;
    
    cblas_dcopy(n, b, 1, r, 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, a, (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, x, 1.0, r);
    
    return ;
}

// Calculate the squared norm of residue
double calc_res_norm2(int size, sparse_matrix_t mat, double* vec, double* sol, double* work)
{
    int n = size;
    double* b = vec, * x = sol;
    sparse_matrix_t a = mat;
    double* r = work;

    calc_res(n, a, b, x, r);
    double rho = cblas_ddot(n, r, 1, r, 1);
    
    return rho;
}

// Calculate the lower part of a banded matrix
void calc_lower_band(int size, double* data, int* ind, int* ptr, double* band, int width)
{
    int n = size, w = width;

    for (int i = 0; i < n * (w+1); i++)
        band[i] = 0.0;

    int pos = *ptr;
    ptr++;
    for (int i = 0; i < n; i++)
    {
        for (; pos < *ptr; pos++)
            if (ind[pos] > i)
                break;
            else
                band[(w+1) * ind[pos] + i - ind[pos]] = data[pos];
        pos = *ptr;
        ptr++;
    }

    return ;
}

// Solve a system using Cholesky decomposition
int evolve_cho(int size, double* data, int* ind, int* ptr, double* init, int width, int rep)
{
    int n = size, w = width;
    sparse_matrix_t a;
    mkl_sparse_d_create_csr(&a, SPARSE_INDEX_BASE_ZERO, n, n, ptr, ptr+1, ind, data);
    double* u = init;
    double* l = malloc((width+1) * n * sizeof(double));
    
    calc_lower_band(n, data, ind, ptr, l, w);

    LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L', n, w, l, w+1);

    int ctr = 0;
    for (int i = 0; i < rep; i++)
    {
        cblas_dtbsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, w, l, w+1, u, 1);
        cblas_dtbsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, n, w, l, w+1, u, 1);

        ctr++;
    }

    free(l);
    mkl_sparse_destroy(a);

    return ctr;
}

// Perform a Gauss-Seidel iteration step
void step_gs(int size, sparse_matrix_t mat, double* vec, double* init)
{
    int n = size;
    double* data;
    int* ind, * ptrb, * ptre;
    int temp;
    mkl_sparse_d_export_csr(mat, &temp, &temp, &temp, &ptrb, &ptre, &ind, &data);

    for (int i = 0; i < n; i++)
    {
        double acc = 0.0, coe;
        for (int pos = *ptrb; pos < *ptre; pos++)
            if (ind[pos] != i)
                acc += data[pos] * init[ind[pos]];
            else
                coe = data[pos];
        init[i] = (vec[i] - acc) / coe;
        ptrb++, ptre++;
    }
    
    return ;
}

// Solve a system using Gauss-Seidel
int solve_gs(int size, sparse_matrix_t mat, double* vec, double* init, double eps, int chk, double* work)
{
    int n = size;
    double* b = vec, * x = init;
    sparse_matrix_t a = mat;
    double* r = work;

    double rho_0 = calc_res_norm2(n, a, b, x, r);

    int ctr = 0;
    while (1)
    {
        step_gs(n, a, b, x);

        ctr++;
        if (ctr % chk == 0)
        {
            double rho = calc_res_norm2(n, a, b, x, r);
            if (rho < rho_0 * eps * eps)
                break;
        }
    }

    return ctr;
}

// Evolve a PDE using Gauss-Seidel
int evolve_gs(int size, double* data, int* ind, int* ptr, double* init, double eps, int chk, int rep)
{
    int n = size;
    sparse_matrix_t a;
    mkl_sparse_d_create_csr(&a, SPARSE_INDEX_BASE_ZERO, n, n, ptr, ptr+1, ind, data);
    double* u = init;
    double* w = malloc(2*n * sizeof(double));
    double* t = w + n;

    int ctr = 0;
    for (int i = 0; i < rep; i++)
    {
        cblas_dcopy(n, u, 1, t, 1);
        ctr += solve_gs(n, a, t, u, eps, chk, w);
    }

    free(w);
    mkl_sparse_destroy(a);

    return ctr;
}

// Solve a system using steepest descent
int solve_sd(int size, sparse_matrix_t mat, double* vec, double* init, double eps, double* work)
{
    int n = size;
    double* b = vec, * x = init;
    sparse_matrix_t a = mat;
    double* r = work, * t = work + n;
    
    double rho = calc_res_norm2(n, a, b, x, r);
    double rho_0 = rho;
    
    int ctr = 0;
    while (1)
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, a, (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, r, 0.0, t);
        double alpha = rho / cblas_ddot(n, r, 1, t, 1);
        cblas_daxpy(n, alpha, r, 1, x, 1);
        cblas_daxpy(n, -alpha, t, 1, r, 1);
        rho = cblas_ddot(n, r, 1, r, 1);
        
        ctr++;
        if (rho < rho_0 * eps * eps)
            break;
    }

    return ctr;
}

// Evolve a PDE using steepest descent
int evolve_sd(int size, double* data, int* ind, int* ptr, double* init, double eps, int rep)
{
    int n = size;
    sparse_matrix_t a;
    mkl_sparse_d_create_csr(&a, SPARSE_INDEX_BASE_ZERO, n, n, ptr, ptr+1, ind, data);
    double* u = init;
    double* w = malloc(3*n * sizeof(double));
    double* t = w + (2*n);

    int ctr = 0;
    for (int i = 0; i < rep; i++)
    {
        cblas_dcopy(n, u, 1, t, 1);
        ctr += solve_sd(n, a, t, u, eps, w);
    }

    free(w);
    mkl_sparse_destroy(a);

    return ctr;
}

// Solve a system using conjugate descent
int solve_cg(int size, sparse_matrix_t mat, double* vec, double* init, double eps, double* work)
{
    int n = size;
    double* b = vec, * x = init;
    sparse_matrix_t a = mat;
    double* r = work, * p = work + n, * t = work + (2*n);
    
    double rho = calc_res_norm2(n, a, b, x, r);
    double rho_0 = rho, rho_old;
    
    int ctr = 0;
    while (1)
    {
        if (!ctr)
            cblas_dcopy(n, r, 1, p, 1);
        else
        {
            double beta = rho / rho_old;
            cblas_dscal(n, beta, p, 1);
            cblas_daxpy(n, 1.0, r, 1, p, 1);
        }

        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, a, (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, p, 0.0, t);

        double alpha = rho / cblas_ddot(n, p, 1, t, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, t, 1, r, 1);

        rho_old = rho;
        rho = cblas_ddot(n, r, 1, r, 1);
        
        ctr++;
        if (rho < rho_0 * eps * eps)
            break;
    }

    return ctr;
}

// Evolve a PDE using conjugate gradient
int evolve_cg(int size, double* data, int* ind, int* ptr, double* init, double eps, int rep)
{
    int n = size;
    sparse_matrix_t a;
    mkl_sparse_d_create_csr(&a, SPARSE_INDEX_BASE_ZERO, n, n, ptr, ptr+1, ind, data);
    double* u = init;
    double* w = malloc(4*n * sizeof(double));
    double* t = w + (3*n);

    int ctr = 0;
    for (int i = 0; i < rep; i++)
    {
        cblas_dcopy(n, u, 1, t, 1);
        ctr += solve_cg(n, a, t, u, eps, w);
    }

    free(w);
    mkl_sparse_destroy(a);

    return ctr;
}

// Perform a multigrid step
void step_mg(int size, sparse_matrix_t* mats, sparse_matrix_t* ints, double* vec, double* init, double* work, int* num, int depth)
{
    int n = size;
    sparse_matrix_t* as = mats, * is = ints;
    double* b = vec, * x = init;
    double* r = work;
    int s = (n-1) * (n-1);

    for (int i = 0; i < *num; i++)
        step_gs(s, as[0], b, x);
    
    if (depth > 1)
    {
        cblas_dcopy(s, b, 1, r, 1);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, as[0], (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, x, 1.0, r);
        mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, is[0], (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, r, 0.0, b + s);
        for (int i = 0; i < (n/2-1) * (n/2-1); i++)
            (x + s)[i] = 0.0;
        step_mg(n/2, as+1, is+1, b + s, x + s, r + s, num+1, depth-1);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, is[0], (struct matrix_descr){SPARSE_MATRIX_TYPE_GENERAL, 0, 0}, x + s, 1.0, x);
    }

    return ;
}

// Solve a system using multigrid
int solve_mg(int size, sparse_matrix_t* mats, sparse_matrix_t* ints, double* vec, double* init, double* work, int* num, int depth, double eps, int chk)
{
    int n = size;
    sparse_matrix_t* as = mats, * is = ints;
    double* b = vec, * x = init;
    double* r = work;
    int s = (n-1) * (n-1);
    
    double rho_0 = calc_res_norm2(s, as[0], b, x, r);

    int ctr = 0;
    while (1)
    {
        step_mg(n, as, is, b, x, r, num, depth);

        ctr++;
        if (ctr % chk == 0)
        {
            double rho = calc_res_norm2(s, as[0], b, x, r);
            if (rho < rho_0 * eps * eps)
                break;
        }
    }

    return ctr;
}

// Evolve a PDE using multigrid
int evolve_mg(int size, double** datas, int** inds, int** ptrs, double* init, int* num, int depth, double eps, int chk, int rep)
{
    int n = size;
    double* x = init;
    sparse_matrix_t* as = malloc(depth * sizeof(sparse_matrix_t)), * is = malloc((depth-1) * sizeof(sparse_matrix_t));
    double* bp, * xp, * rp;
    int s = (n-1) * (n-1);

    int acc = 0, temp = n;
    for (int i = 0; i < depth; i++)
    {
        acc += (temp-1) * (temp-1);
        temp /= 2;
    }
    bp = malloc(acc * sizeof(double)), xp = malloc(acc * sizeof(double)), rp = malloc(acc * sizeof(double));

    temp = n;
    mkl_sparse_d_create_csr(as, SPARSE_INDEX_BASE_ZERO, s, s, ptrs[0], ptrs[0]+1, inds[0], datas[0]);
    for (int i = 1; i < depth; i++)
    {
        mkl_sparse_d_create_csr(is + (i-1), SPARSE_INDEX_BASE_ZERO, (temp-1) * (temp-1), (temp/2-1) * (temp/2-1), ptrs[i], ptrs[i]+1, inds[i], datas[i]);
        sparse_matrix_t t;
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, as[i-1], is[i-1], &t);
        mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE, is[i-1], t, as + i);
        mkl_sparse_destroy(t);
        temp /= 2;
    }
    
    cblas_dcopy(s, x, 1, xp, 1);
    
    int ctr = 0;
    for (int i = 0; i < rep; i++)
    {
        cblas_dcopy(s, xp, 1, bp, 1);
        ctr += solve_mg(n, as, is, bp, xp, rp, num, depth, eps, chk);
    }
    
    cblas_dcopy(s, xp, 1, x, 1);

    for (int i = 1; i < depth; i++)
    {
        mkl_sparse_destroy(as[i]);
        mkl_sparse_destroy(is[i-1]);
    }
    mkl_sparse_destroy(as[0]);
    free(bp), free(xp), free(rp);
    free(as), free(is);

    return ctr;
}
