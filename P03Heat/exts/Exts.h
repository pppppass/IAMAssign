#include <stdlib.h>
#include <mkl.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

void calc_res(int size, sparse_matrix_t mat, double* vec, double* sol, double* res);
double calc_res_norm2(int size, sparse_matrix_t mat, double* vec, double* sol, double* work);
void calc_lower_band(int size, double* data, int* ind, int* ptr, double* band, int width);

int evolve_cho(int size, double* data, int* ind, int* ptr, double* init, int width, int rep);

void step_gs(int size, sparse_matrix_t mat, double* vec, double* init);
int solve_gs(int size, sparse_matrix_t mat, double* vec, double* init, double eps, int chk, double* work);
int evolve_gs(int size, double* data, int* ind, int* ptr, double* init, double eps, int chk, int rep);

int solve_sd(int size, sparse_matrix_t mat, double* vec, double* init, double eps, double* work);
int evolve_sd(int size, double* data, int* ind, int* ptr, double* init, double eps, int rep);

int solve_cg(int size, sparse_matrix_t mat, double* vec, double* init, double eps, double* work);
int evolve_cg(int size, double* data, int* ind, int* ptr, double* init, double eps, int rep);

void step_mg(int size, sparse_matrix_t* mats, sparse_matrix_t* ints, double* vec, double* init, double* work, int* num, int depth);
int solve_mg(int size, sparse_matrix_t* mats, sparse_matrix_t* ints, double* vec, double* init, double* work, int* num, int depth, double eps, int chk);
int evolve_mg(int size, double** datas, int** inds, int** ptrs, double* init, int* num, int depth, double eps, int chk, int rep);
