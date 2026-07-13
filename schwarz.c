/*
 * schwarz.c — metodo di Schwarz additivo algebrico (MPI), senza
 * sovrapposizione (= Jacobi a blocchi), con l'algoritmo di Thomas
 * come solutore locale esatto.
 *
 * Iterazione (Richardson precondizionata):
 *   u^{m} = u^{m-1} + sum_i R_i^T A_i^{-1} R_i (b - A u^{m-1})
 *
 * dove A_i = R_i A R_i^T è il blocco diagonale n_loc x n_loc del processo i. 
 * Per il problema di Poisson ogni A_i è tridiagonale:
 * il sistema locale A_i w = r_loc viene risolto con thomas_solve,
 * interamente in locale e senza comunicazioni.
 *
 * Gli operatori R_i e R_i^T non sono mai costruiti esplicitamente:
 * R_i corrisponde all'accesso alla porzione locale dei vettori,
 * R_i^T all'aggiornamento delle sole componenti di competenza.
 *
 * Riferimenti: 
 * Dolean, Jolivet, Nataf, "An Introduction to Domain Decomposition Methods", SIAM 2015, Sez. 1.2-1.3; 
 * Quarteroni et al., Sez. 4.2.4 (Jacobi a blocchi) e 3.7.1 (Thomas).
 */

#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "solver.h"

int MPI_Schwarz(const double *A_local, const double *b_local,
                double *x, int N, int n_loc, int rank,
                double tol, int max_iter)
{
    /* Estrazione del blocco tridiagonale locale A_i 
     * La riga locale i è la riga globale g = rank*n_loc + i; il blocco
     * diagonale occupa le colonne globali [rank*n_loc, rank*n_loc + n_loc).
     * Nota: per una matrice qualunque questa è un'approssimazione
     * tridiagonale del blocco; per la matrice di Poisson il blocco È
     * esattamente tridiagonale, quindi il solutore locale è esatto. */
    double *low = malloc((size_t)n_loc * sizeof(double));
    double *diag = malloc((size_t)n_loc * sizeof(double));
    double *up = malloc((size_t)n_loc * sizeof(double));
    double *r_loc = malloc((size_t)n_loc * sizeof(double));
    double *w = malloc((size_t)n_loc * sizeof(double));
    double *x_new_local = malloc((size_t)n_loc * sizeof(double));
    double *beta = malloc((size_t)n_loc * sizeof(double));
    double *y = malloc((size_t)n_loc * sizeof(double));

    const int offset = rank * n_loc;   /* prima colonna del blocco */
    for (int i = 0; i < n_loc; i++) {
        const double *row = &A_local[(size_t)i * N];
        diag[i] = row[offset + i];
        low[i] = (i > 0) ? row[offset + i - 1] : 0.0;
        up[i] = (i < n_loc - 1) ? row[offset + i + 1] : 0.0;
    }

    /* ||b||_2 globale */
    double nb_loc = 0.0, nb;
    for (int i = 0; i < n_loc; i++) nb_loc += b_local[i] * b_local[i];
    MPI_Allreduce(&nb_loc, &nb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    nb = sqrt(nb);

    int iter;
    for (iter = 1; iter <= max_iter; iter++) {

        /* 1) residuo locale: r_loc = R_i (b - A u) */
        double nr_loc = 0.0, nr;
        for (int i = 0; i < n_loc; i++) {
            const double *row = &A_local[(size_t)i * N];
            double r = b_local[i];
            for (int j = 0; j < N; j++) r -= row[j] * x[j];
            r_loc[i] = r;
            nr_loc += r * r;
        }

        /* criterio d'arresto sul residuo appena calcolato */
        MPI_Allreduce(&nr_loc, &nr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (sqrt(nr) / nb <= tol) { iter--; break; }

        /* 2) soluzione locale esatta con Thomas: A_i w = r_loc */
        thomas_solve(n_loc, low, diag, up, r_loc, w, beta, y);

        /* 3) aggiornamento u^{m} = u^{m-1} + R_i^T w e assemblaggio */
        for (int i = 0; i < n_loc; i++)
            x_new_local[i] = x[offset + i] + w[i];

        MPI_Allgather(x_new_local, n_loc, MPI_DOUBLE, x, n_loc, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    free(low); free(diag); free(up);
    free(r_loc); free(w); free(x_new_local);
    free(beta); free(y);
    return iter;
}