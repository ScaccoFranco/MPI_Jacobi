#ifndef JACOBI_H
#define JACOBI_H

void genera_matrici(double *A, double *b, const int n, unsigned int seed);

void genera_matrici_diag_dom(double *A, double *b, const int n, unsigned int seed); //genera matrice A dominante per diagonale (converge jacobi)

void genera_matrici_diagdom_xones(double *A, double *b, const int n, unsigned int seed); // come funz prima ma genera in modo che x = unitaria (x = ones(size, 1))

void fill_const(double *A, const double val, const int rows, const int cols);

/*
Funzione per generare matrici in modo randomico per sistema lineare A x = b
A: n x n
b: n x 1

una rand e una diag dominante per righe (converge jacobi)
*/

double* jacobi(const double* A, const double* b, const double* x0, const int size, const double toll, const unsigned int itermax);


double* MPI_Jacobi(const double* A, const double* b, const double* x0, const int size, const double toll, const unsigned int itermax);

/*
Funizione con metodo di jacobi:
soluzione di un problema lineare del tipo Ax = b
quindi size la dimensione di b o il numeri di righe o colonne di A
const double* A: matrice size x size
const double* b: vettore size x 1 
const double* x0: vettore size x 1 da cui partire per il calcolo 
toll è la tolleranza minima dopo la quale il programma si ferma
itermax numero massimo di iterazioni
*/






#endif