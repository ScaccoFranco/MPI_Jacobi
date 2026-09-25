/*
 * main.c — programma principale: problema di Poisson 1D risolto con
 * Jacobi o con Schwarz additivo (+ Thomas), in parallelo con MPI.
 *
 * Uso: mpirun -np p ./solver <N> <jacobi|schwarz> [tol] [max_iter]
 *   N         numero di nodi interni (multiplo di p)
 *   tol       tolleranza sul residuo relativo (default 1e-6)
 *   max_iter  numero massimo di iterazioni (default 10000000)
 *
 * Passi:
 *   1. il rank 0 assembla A (N x N) e b con poisson_generate;
 *   2. MPI_Scatter distribuisce a ogni processo le sue n_loc righe di A e di b;
 *   3. si chiama il solutore scelto, misurando solo il suo tempo;
 *   4. il rank 0 stampa iterazioni, tempo (del processo piu' lento) ed errore
 *      rispetto alla soluzione esatta.
 *
 * Nota: ricostruito sull'interfaccia e sull'output dell'eseguibile build/solver,
 * perche' il main.c originale di questa versione non e' nel repository.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "solver.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <N> <jacobi|schwarz> [tol] [max_iter]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const int N = atoi(argv[1]);
    const char *metodo = argv[2];
    const double tol = (argc > 3) ? atof(argv[3]) : 1e-6;
    const int max_iter = (argc > 4) ? atoi(argv[4]) : 10000000;
    const int usa_jacobi = (strcmp(metodo, "jacobi") == 0);

    /* Tutti i processi fanno gli stessi controlli, quindi escono tutti insieme. */
    if (N < size || N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Errore: N=%d non e' un multiplo positivo di p=%d\n", N, size);
        MPI_Finalize();
        return 1;
    }
    if (!usa_jacobi && strcmp(metodo, "schwarz") != 0) {
        if (rank == 0)
            fprintf(stderr, "Metodo sconosciuto: %s (usare jacobi o schwarz)\n", metodo);
        MPI_Finalize();
        return 1;
    }

    const int n_loc = N / size;
    double *A_local = malloc((size_t)n_loc * N * sizeof(double));
    double *b_local = malloc((size_t)n_loc * sizeof(double));
    double *x = calloc((size_t)N, sizeof(double));   /* innesco x^(0) = 0 */
    double *A = NULL, *b = NULL;

    /* Solo il rank 0 assembla il problema completo, poi lo distribuisce:
     * il processo r riceve le righe r*n_loc .. r*n_loc + n_loc - 1. */
    if (rank == 0) {
        A = malloc((size_t)N * N * sizeof(double));
        b = malloc((size_t)N * sizeof(double));
        poisson_generate(N, A, b);
    }
    MPI_Scatter(A, n_loc * N, MPI_DOUBLE, A_local, n_loc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, n_loc, MPI_DOUBLE, b_local, n_loc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Misuro solo il solutore: generazione e distribuzione restano fuori. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    int iter = usa_jacobi
        ? MPI_Jacobi (A_local, b_local, x, N, n_loc, rank, tol, max_iter)
        : MPI_Schwarz(A_local, b_local, x, N, n_loc, rank, tol, max_iter);
    double t = MPI_Wtime() - t0, t_max;

    /* Il tempo che conta e' quello del processo piu' lento. */
    MPI_Reduce(&t, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* x e' completo e uguale su tutti i processi: basta il rank 0 per l'errore. */
    if (rank == 0) {
        printf("metodo=%-8s N=%-7d p=%-3d iterazioni=%-8d tempo=%.4f s  err_max=%.3e\n",
               metodo, N, size, iter, t_max, poisson_max_error(N, x));
        if (iter > max_iter)
            printf("attenzione: tolleranza non raggiunta in %d iterazioni\n", max_iter);
        free(A);
        free(b);
    }

    free(A_local);
    free(b_local);
    free(x);
    MPI_Finalize();
    return 0;
}
