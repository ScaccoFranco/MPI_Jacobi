/*
 * schwarz.c
 *
 * Qui c'è il metodo di Schwarz additivo in versione algebrica, parallelizzato
 * con MPI. Ogni processo si prende delta nodi in più per lato dai vicini
 * (sovrapposizione) e uso la variante RAS (Restricted Additive Schwarz).
 * Come solutore locale uso l'algoritmo di Thomas, che risolve esattamente.
 *
 * L'iterazione è una Richardson precondizionata:
 *   u^{m} = u^{m-1} + sum_i R_i^T W_i A_i^{-1} R_i (b - A u^{m-1})
 *
 * Cosa sono i vari pezzi:
 * - I_i sono le righe del processo i: le sue n_loc più delta righe per lato
 *   prese dai vicini. Dal lato del bordo fisico non aggiungo niente.
 * - A_i = R_i A R_i^T è il pezzo di A con righe e colonne in I_i. Per Poisson
 *   è ancora tridiagonale, quindi A_i w = r_loc lo risolvo con Thomas, tutto
 *   in locale. A_i resta sempre la stessa, quindi la fattorizzo una volta sola
 *   prima del ciclo e dentro faccio solo le due sostituzioni.
 * - W_i vale 1 sulle righe mie e 0 su quelle prese dai vicini. In pratica
 *   ogni processo aggiorna solo le sue incognite, così nessuna incognita viene
 *   aggiornata due volte.
 *
 * Se delta = 0 gli I_i non si sovrappongono, W_i = I e torna il Jacobi a
 * blocchi.
 *
 * Per il residuo sulle righe estese non serve tutto il vettore: bastano le
 * celle di overlap che scambio con i vicini (MPI_Sendrecv, delta + 1 valori
 * per lato).
 *
 * R_i, R_i^T e W_i non li costruisco mai come matrici: R_i è semplicemente
 * leggere le righe estese, R_i^T W_i è aggiornare solo le componenti mie.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "solver.h"

/* Allunga un vettore (low, diag, up o b) sulle righe di I_i:
 * v_est[0 .. dl-1]           le ultime dl righe del vicino di sinistra,
 * v_est[dl .. dl+n_loc-1]    le mie righe,
 * v_est[dl+n_loc .. m-1]     le prime dr righe del vicino di destra.
 * Lo chiamo solo una volta, prima del ciclo. */
static void estendi(const double *v, double *v_est, int n_loc, int dl, int dr,
                    int sx, int dx)
{
    memcpy(&v_est[dl], v, (size_t)n_loc * sizeof(double));
    /* mando le mie prime righe a sinistra, ricevo quelle di destra */
    MPI_Sendrecv(&v[0],              dl, MPI_DOUBLE, sx, 2,
                 &v_est[dl + n_loc], dr, MPI_DOUBLE, dx, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* mando le mie ultime righe a destra, ricevo quelle di sinistra */
    MPI_Sendrecv(&v[n_loc - dr],     dr, MPI_DOUBLE, dx, 3,
                 &v_est[0],          dl, MPI_DOUBLE, sx, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int MPI_Schwarz(const double *low, const double *diag, const double *up,
                const double *b, double *u, int n_loc, int rank, int size,
                double tol, int max_iter, int delta)
{
    const int sx = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    const int dx = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* quante righe prendo da sinistra e da destra (0 se sono al bordo) */
    const int dl = (rank > 0)        ? delta : 0;
    const int dr = (rank < size - 1) ? delta : 0;
    const int m = dl + n_loc + dr;      /* dimensione di A_i */
    const int larg = delta + 1;         /* celle di overlap per lato */

    double *low_est  = malloc((size_t)m * sizeof(double));
    double *diag_est = malloc((size_t)m * sizeof(double));
    double *up_est   = malloc((size_t)m * sizeof(double));
    double *b_est    = malloc((size_t)m * sizeof(double));
    estendi(low,  low_est,  n_loc, dl, dr, sx, dx);
    estendi(diag, diag_est, n_loc, dl, dr, sx, dx);
    estendi(up,   up_est,   n_loc, dl, dr, sx, dx);
    estendi(b,    b_est,    n_loc, dl, dr, sx, dx);

    /* ue sono le incognite con larg celle di overlap per lato; le mie sono in
     * ue[larg .. larg+n_loc-1]. Uso calloc così ai bordi fisici resta 0. */
    double *ue = calloc((size_t)n_loc + 2 * (size_t)larg, sizeof(double));
    memcpy(&ue[larg], &u[1], (size_t)n_loc * sizeof(double));
    const int off = larg - dl;          /* dove sta in ue la riga estesa 0 */

    double *r_loc = malloc((size_t)m * sizeof(double));
    double *w     = malloc((size_t)m * sizeof(double));
    double *alpha = malloc((size_t)m * sizeof(double));
    double *beta  = malloc((size_t)m * sizeof(double));
    double *y     = malloc((size_t)m * sizeof(double));

    /* Fattorizzo A_i una volta sola. Thomas usa solo low_est[1..m-1] e
     * up_est[0..m-2]; low_est[0] e up_est[m-1] sono i legami con i nodi fuori
     * da I_i e qui non servono. */
    thomas_factor(m, low_est, diag_est, up_est, beta, alpha);

    /* ||b||_2 globale, contando solo le righe mie */
    double nb_loc = 0.0, nb;
    for (int i = 0; i < n_loc; i++) nb_loc += b[i] * b[i];
    MPI_Allreduce(&nb_loc, &nb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    nb = sqrt(nb);

    /* a inizio giro k in ue c'è u^{(k)} */
    int k;
    for (k = 0; ; k++) {

        /* 1) residuo sulle righe estese: r_loc = R_i (b - A u) */
        bordi_scambia(ue, n_loc, larg, sx, dx);
        double nr_loc = 0.0, nr;
        for (int j = 0; j < m; j++) {
            const int i = off + j;
            double r = b_est[j] - low_est[j] * ue[i - 1] - diag_est[j] * ue[i]
                       - up_est[j] * ue[i + 1];
            r_loc[j] = r;
            /* nella norma metto solo le righe mie, le altre le conta il vicino */
            if (j >= dl && j < dl + n_loc)
                nr_loc += r * r;
        }

        /* controllo se mi posso fermare, con il residuo appena calcolato */
        MPI_Allreduce(&nr_loc, &nr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (sqrt(nr) / nb <= tol) break;      /* la soluzione è u^{(k)} */

        /* come in Jacobi controllo anche l'ultima iterata u^{(max_iter)} */
        if (k == max_iter) { k = max_iter + 1; break; }   /* tolleranza non raggiunta */

        /* 2) risolvo il problema locale A_i w = r_loc con Thomas */
        thomas_solve_factored(m, beta, alpha, up_est, r_loc, w, y);

        /* 3) aggiorno u^{m} = u^{m-1} + R_i^T W_i w, solo sulle mie incognite */
        for (int i = 0; i < n_loc; i++)
            ue[larg + i] += w[dl + i];
    }

    /* rimetto la soluzione nel vettore di chi mi ha chiamato */
    memcpy(&u[1], &ue[larg], (size_t)n_loc * sizeof(double));

    free(low_est); free(diag_est); free(up_est); free(b_est); free(ue);
    free(r_loc); free(w);
    free(alpha); free(beta); free(y);
    return k;
}
