/*
 * main.c
 *
 * Programma principale: risolvo il problema di Poisson 1D in parallelo con
 * MPI, con Jacobi oppure con Schwarz additivo (+ Thomas).
 *
 * Uso: mpirun -np p ./solver <N> <jacobi|schwarz> [tol] [max_iter] [delta]
 *   N         numero di nodi interni (multiplo di p)
 *   tol       tolleranza sul residuo relativo (default 1e-6)
 *   max_iter  numero massimo di iterazioni (default 10000000)
 *   delta     solo per schwarz: nodi di sovrapposizione per lato (default 0,
 *             cioe' Jacobi a blocchi); serve delta + 1 <= N/p
 *
 * Cosa fa:
 *   1. ogni processo si genera da solo le sue n_loc righe di A (tre
 *      diagonali) e di b con poisson_generate_local, quindi non c'e' una
 *      matrice globale e non serve MPI_Scatter;
 *   2. chiamo il solutore scelto e misuro solo il suo tempo;
 *   3. il rank 0 stampa iterazioni, tempo (quello del processo piu' lento) ed
 *      errore rispetto alla soluzione esatta, messo insieme con MPI_Reduce
 *      (MPI_MAX).
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
            fprintf(stderr, "Uso: %s <N> <jacobi|schwarz> [tol] [max_iter] [delta]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const int N = atoi(argv[1]);
    const char *metodo = argv[2];
    const double tol = (argc > 3) ? atof(argv[3]) : 1e-6;
    const int max_iter = (argc > 4) ? atoi(argv[4]) : 10000000;
    const int delta = (argc > 5) ? atoi(argv[5]) : 0;
    const int usa_jacobi = (strcmp(metodo, "jacobi") == 0);

    /* I controlli li fanno tutti i processi, cosi' se c'e' un errore escono tutti insieme. */
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
    if (!usa_jacobi && (delta < 0 || delta + 1 > n_loc)) {
        if (rank == 0)
            fprintf(stderr, "Errore: serve 0 <= delta <= N/p - 1 = %d\n", n_loc - 1);
        MPI_Finalize();
        return 1;
    }
    double *low  = malloc((size_t)n_loc * sizeof(double));
    double *diag = malloc((size_t)n_loc * sizeof(double));
    double *up   = malloc((size_t)n_loc * sizeof(double));
    double *b    = malloc((size_t)n_loc * sizeof(double));
    double *u    = calloc((size_t)n_loc + 2, sizeof(double));  /* innesco u^(0) = 0, celle di overlap 0 */

    /* Ogni processo si genera le righe r*n_loc .. r*n_loc + n_loc - 1. */
    poisson_generate_local(N, n_loc, rank, low, diag, up, b);

    /* Misuro solo il solutore, la generazione la lascio fuori. */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    int iter = usa_jacobi
        ? MPI_Jacobi (low, diag, up, b, u, n_loc, rank, size, tol, max_iter)
        : MPI_Schwarz(low, diag, up, b, u, n_loc, rank, size, tol, max_iter, delta);
    double t = MPI_Wtime() - t0, t_max;

    /* Come tempo prendo quello del processo piu' lento. */
    MPI_Reduce(&t, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Ogni processo ha solo le sue incognite, quindi l'errore massimo lo metto insieme con MPI_MAX. */
    double err_loc = poisson_max_error_local(N, n_loc, rank, &u[1]), err;
    MPI_Reduce(&err_loc, &err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("metodo=%-8s N=%-7d p=%-3d iterazioni=%-8d tempo=%.4f s  err_max=%.3e",
               metodo, N, size, iter, t_max, err);
        if (!usa_jacobi)
            printf("  delta=%d", delta);
        printf("\n");
        if (iter > max_iter)
            printf("attenzione: tolleranza non raggiunta in %d iterazioni\n", max_iter);
    }

    free(low); free(diag); free(up); free(b); free(u);
    MPI_Finalize();
    return 0;
}
