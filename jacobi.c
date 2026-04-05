#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>
#include "jacobi.h"


/*
    x generare numeri "pseudo-casuali" uso algoritmo
    Generatore lineare congruenziale (LCG):
    formula ricorsiva: X(n+1) = (a*X(n) + c) mod m
    con:
    X(n) - n esimo elemento della sucessione pseudocasuale
    m, m > 0 - il modulo
    a, 0 < a < m - il moltiplicatore
    c, 0 <= c < m - l'incremento
    X(0), 0 <= X(0) < m - il seme, o valore iniziale


    valori usati:
    a = 1664525 - moltiplicatore usato da Knuth, The Art of Computer Programming
    c = 1013904223 - incremento scelto da Numerical Recipes
    m = 2^32 - modulo implicito lavorando con unsigned int a 32 bit 
*/

static double rand_double(unsigned int *seed) {
    double max_val = 2.0;

    //ogni volta che viene chaiamata la funz cambia il seed (state)
    *seed = (*seed * 1664525u + 1013904223u); // LCG classico 

    // genero rand tra [0,1] e poi passo a [-max, max]
    return ((double)(*seed) / (double)UINT32_MAX) * 2.0 * max_val - max_val;
}



// funzione per generare matrici A, b random
void genera_matrici(double *A, double *b, const int n, unsigned int seed) {
    unsigned int seme = seed;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = rand_double(&seme); // assegna valore e cambia seed
        }
        b[i] = rand_double(&seme); // pure qua
    }
}


// genero matrice diagonale dominante per righe
void genera_matrici_diag_dom(double *A, double *b, const int n, unsigned int seed) {
    unsigned int seme = seed;
    double margine = 1.0; // margine di dominanza

    for (int i = 0; i < n; i++) {
        double somma_righe = 0.0;

        // faccio prima elementi non diagonali
        for (int j = 0; j < n; j++) {
            if (j != i) {
                A[i * n + j] = rand_double(&seme);
                somma_righe += fabs(A[i * n + j]); //  fabs = abs per double, math.h
            }
        }

        // per la diagonale tengo la somma delle righe 
        A[i * n + i] = somma_righe + margine;

        b[i] = rand_double(&seme);
    }
}

 // come funz prima ma genera in modo che x = unitaria (x = ones(size, 1))
void genera_matrici_diagdom_xones(double *A, double *b, const int n, unsigned int seed) {
    unsigned int seme = seed;
    double margine = 1.0; // margine di dominanza

    for (int i = 0; i < n; i++) {
        double somma_righe = 0.0;
        b[i] = 0.0;

        // faccio prima elementi non diagonali
        for (int j = 0; j < n; j++) {
            if (j != i) {
                A[i * n + j] = rand_double(&seme);
                somma_righe += fabs(A[i * n + j]); //  fabs = abs per double, math.hb[i] += A[i * n + j]; // i valori di b sono la sommatoria delle righe di A
                b[i] += A[i * n + j]; // i valori di b sono la sommatoria delle righe di A
            }
        }

        // per la diagonale tengo la somma delle righe 
        A[i * n + i] = somma_righe + margine;
        b[i] += A[i * n + i]; // i valori di b sono la sommatoria delle righe di A
    }
}

// la uso per zeros() di matlab
void fill_const(double *A, const double val, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i*cols + j] = val;
        }
    }
}



// VERSIONE SENZA MPI!!
double* jacobi(const double* A, const double* b, const double* x0, const int size, const double toll, const unsigned int itermax) {
    /*
    JACOBI: Xi = (bi - sommatoria i!=j di Aij*Xj)/Aii
    */

    double diff;
    double *x = malloc(size * sizeof(double));
    double *x_new = malloc(size * sizeof(double));
    memcpy(x, x0, size*sizeof(double));

    for (int i = 0; i < itermax; i++) {
        diff = 0.0;

        // singola iterazione
        for (int j = 0; j < size; j++) {
            double sommatoria = 0.0;

            for (int k = 0; k < size; k++) {
                if (k != j) {
                    sommatoria += A[j * size + k] * x[k];
                }
            }

            x_new[j] = (b[j] - sommatoria) / A[j*size + j]; // formula iterativa jacobi 
        }

        // calcolo toll
        for (int j = 0; j < size; j++) {
            // diff è il massimo scarto, norma infinito!
            diff = fmax(diff, fabs(x_new[j] - x[j]));
            x[j] = x_new[j]; 
        }

        if(diff < toll) {
            break;
        }
    }

    return x;
}

// VERSIONE SENZA MPI!!
double* MPI_Jacobi(const double* A, const double* b, const double* x0, const int size, const double toll, const unsigned int itermax) {
    /*
    JACOBI: Xi = (bi - sommatoria i!=j di Aij*Xj)/Aii

    con MPI: divido la matrice in blocchi di righe e li assegno ai diversi processi

    */

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    //Righe assegnate a questo processo
    int per_proc = size / world_size;
    int row_start = rank * per_proc;
    int row_end = row_start + per_proc;



    double local_diff, global_diff;
    double *x = malloc(size * sizeof(double));
    double *x_new = malloc(size * sizeof(double));
    memcpy(x, x0, size*sizeof(double));

    for (int i = 0; i < itermax; i++) {
        local_diff = 0.0;

        // singola iterazione (divisa nei vari processi)
        for (int j = row_start; j < row_end; j++) { // verificare se fare < o <=
            double sommatoria = 0.0;

            for (int k = 0; k < size; k++) {
                if (k != j) {
                    sommatoria += A[j * size + k] * x[k];
                }
            }

            x_new[j] = (b[j] - sommatoria) / A[j*size + j]; // formula iterativa jacobi 
        }

        // calcolo toll
        for (int j = row_start; j < row_end; j++) { // verifica pure qua
            // diff è il massimo scarto, norma infinito!
            local_diff = fmax(local_diff, fabs(x_new[j] - x[j]));
            x[j] = x_new[j]; 
        }


        // contivido x_new su tutti i processi per poter fare l'iterazione successiva. a tutti serve quindi uso Allghater invece di gather
        MPI_Allgather(x_new + row_start, per_proc, MPI_DOUBLE, x, per_proc, MPI_DOUBLE, MPI_COMM_WORLD);

        // pure qua serve a tutti quindi uso Allreduce e non reduce, altrimenti posso usare gather e poi broadcast?
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (global_diff < toll) {
            break;
        }
    }

    return x;
}
