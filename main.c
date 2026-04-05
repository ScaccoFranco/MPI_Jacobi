#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "jacobi.h"

int main () {

    // faccio sempre nel main, non nella funzione!!
    MPI_Init(NULL, NULL);
    
    int size = 100;
    unsigned int seed = 1000;


    double toll = 1e-3;
    unsigned int itermax = 1000U;

    double *A = malloc(size * size * sizeof(double)); 
    double *b = malloc(size * sizeof(double)); 
    double *x0 = malloc(size * sizeof(double)); 

    genera_matrici_diagdom_xones(A, b, size, seed);
    fill_const(x0, 0.0, size, 1);

    // double *sol = jacobi(A, b, x0, size, toll, itermax);
    double *sol = MPI_Jacobi(A, b, x0, size, toll, itermax);


    for (int i = 0; i < 5; i++) {
        printf("%.4f\n", sol[i]);
    }

    //pure questo main e non funzione
    MPI_Finalize();

    return 0;
}