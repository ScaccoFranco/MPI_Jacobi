/*
 * schwarz.c — metodo di Schwarz additivo algebrico (MPI), senza
 * sovrapposizione (= Jacobi a blocchi), con l'algoritmo di Thomas
 * come solutore locale esatto.
 *
 * Iterazione (Richardson precondizionata):
 *   u^{m} = u^{m-1} + sum_i R_i^T A_i^{-1} R_i (b - A u^{m-1})
 *
 * dove A_i = R_i A R_i^T è il blocco diagonale n_loc x n_loc del processo i.
 * Per il problema di Poisson ogni A_i è tridiagonale ed è gia' disponibile
 * come tre diagonali (low, diag, up): il sistema locale A_i w = r_loc viene
 * risolto con Thomas, interamente in locale e senza comunicazioni.
 * A_i non cambia tra un'iterazione e l'altra, quindi Thomas la fattorizza una
 * volta prima del ciclo; a ogni iterazione restano solo le due sostituzioni.
 *
 * Il residuo sulle righe locali usa solo le celle di overlap scambiate con i
 * vicini (MPI_Sendrecv), non il vettore globale.
 *
 * Gli operatori R_i e R_i^T non sono mai costruiti esplicitamente:
 * R_i corrisponde all'accesso alla porzione locale dei vettori,
 * R_i^T all'aggiornamento delle sole componenti di competenza.
 */

#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "solver.h"

int MPI_Schwarz(const double *low, const double *diag, const double *up,
                const double *b, double *u, int n_loc, int rank, int size,
                double tol, int max_iter)
{
    const int sx = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    const int dx = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    double *r_loc = malloc((size_t)n_loc * sizeof(double));
    double *w     = malloc((size_t)n_loc * sizeof(double));
    double *alpha = malloc((size_t)n_loc * sizeof(double));
    double *beta  = malloc((size_t)n_loc * sizeof(double));
    double *y     = malloc((size_t)n_loc * sizeof(double));

    /* Fattorizzazione di A_i, una volta sola. Thomas usa solo low[1..n-1] e
     * up[0..n-2]: low[0] e up[n_loc-1] sono gli accoppiamenti con i vicini. */
    thomas_factor(n_loc, low, diag, up, beta, alpha);

    /* ||b||_2 globale */
    double nb_loc = 0.0, nb;
    for (int i = 0; i < n_loc; i++) nb_loc += b[i] * b[i];
    MPI_Allreduce(&nb_loc, &nb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    nb = sqrt(nb);

    /* All'inizio del giro k, u contiene u^{(k)}. */
    int k;
    for (k = 0; k < max_iter; k++) {

        /* 1) residuo locale: r_loc = R_i (b - A u) */
        bordi_scambia(u, n_loc, sx, dx);
        double nr_loc = 0.0, nr;
        for (int i = 1; i <= n_loc; i++) {
            double r = b[i - 1] - low[i - 1] * u[i - 1] - diag[i - 1] * u[i]
                       - up[i - 1] * u[i + 1];
            r_loc[i - 1] = r;
            nr_loc += r * r;
        }

        /* criterio d'arresto sul residuo appena calcolato */
        MPI_Allreduce(&nr_loc, &nr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (sqrt(nr) / nb <= tol) break;

        /* 2) soluzione locale esatta con Thomas: A_i w = r_loc */
        thomas_solve_factored(n_loc, beta, alpha, up, r_loc, w, y);

        /* 3) aggiornamento u^{m} = u^{m-1} + R_i^T w: solo le mie incognite */
        for (int i = 1; i <= n_loc; i++)
            u[i] += w[i - 1];
    }
    if (k == max_iter) k = max_iter + 1;   /* tolleranza non raggiunta */

    free(r_loc); free(w);
    free(alpha); free(beta); free(y);
    return k;
}
