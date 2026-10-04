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
 *
 * Qui ci sono anche bordi_scambia e norma_globale, che usa anche jacobi.c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "solver.h"

/*
 * Scambio delle celle di overlap con i processi vicini (la usa anche jacobi.c).
 *
 * Ogni processo tiene u[0 .. n_loc + 2*larg - 1]. Le sue incognite sono in
 * u[larg .. larg+n_loc-1], in u[0 .. larg-1] ci metto le ultime larg incognite
 * del vicino di sinistra e in u[larg+n_loc .. n_loc+2*larg-1] le prime larg
 * del vicino di destra.
 *
 * Jacobi blocchi usa larg = 1: A e' tridiagonale, quindi per fare A u sulle mie righe
 * mi basta un valore per lato. Schwarz con sovrapposizione delta usa
 * larg = delta + 1, perche' il residuo lo calcola anche sui delta nodi presi
 * dal vicino, e per la loro riga serve un valore in piu'. Prima del ciclo
 * Schwarz la usa anche con larg = delta, per prendersi dai vicini le righe
 * di A e di b che servono per I_i.
 *
 * Agli estremi il vicino e' MPI_PROC_NULL: la send non fa niente e la receive
 * non tocca il buffer. Cosi' le celle di overlap del rank 0 a sinistra e
 * dell'ultimo rank a destra restano a 0, che sono proprio le condizioni al
 * bordo u(0) = u(1) = 0.
 * Uso MPI_Sendrecv cosi' non devo mettere in ordine a mano send e receive
 * per evitare il deadlock.
 */
void bordi_scambia(double *u, int n_loc, int larg, int sx, int dx)
{
    /* mando i miei primi larg valori a sinistra, ricevo le celle di overlap di destra */
    MPI_Sendrecv(&u[larg],         larg, MPI_DOUBLE, sx, 0,
                 &u[larg + n_loc], larg, MPI_DOUBLE, dx, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* mando i miei ultimi larg valori a destra, ricevo le celle di overlap di sinistra */
    MPI_Sendrecv(&u[n_loc],        larg, MPI_DOUBLE, dx, 1,
                 &u[0],            larg, MPI_DOUBLE, sx, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* Norma 2 di un vettore distribuito: ogni processo passa la somma dei
 * quadrati delle sue componenti, la Allreduce le somma e il risultato
 * arriva a tutti. */
double norma_globale(double somma_quadrati_loc)
{
    double somma;
    MPI_Allreduce(&somma_quadrati_loc, &somma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(somma);
}

/* Allunga un vettore (low, diag, up o b) con delta celle per lato e ci mette
 * le righe dei vicini con bordi_scambia. Le righe di I_i partono da
 * v_est[delta - dl]; dal lato del bordo fisico le celle restano a 0 e non
 * vengono usate. Lo chiamo solo prima del ciclo. */
static double *allunga(const double *v, int n_loc, int delta, int sx, int dx)
{
    double *v_est = calloc((size_t)n_loc + 2 * (size_t)delta, sizeof(double));
    memcpy(&v_est[delta], v, (size_t)n_loc * sizeof(double));
    bordi_scambia(v_est, n_loc, delta, sx, dx);
    return v_est;
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

    /* le righe di I_i sono low_est[0 .. m-1] ecc.; s e' dove iniziano nei
     * vettori allungati */
    const int s = delta - dl;
    double *low_a  = allunga(low,  n_loc, delta, sx, dx), *low_est  = low_a  + s;
    double *diag_a = allunga(diag, n_loc, delta, sx, dx), *diag_est = diag_a + s;
    double *up_a   = allunga(up,   n_loc, delta, sx, dx), *up_est   = up_a   + s;
    double *b_a    = allunga(b,    n_loc, delta, sx, dx), *b_est    = b_a    + s;

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
     * da I_i e qui non servono. A_i e' SPD, quindi un pivot nullo vorrebbe
     * dire che c'e' un errore nei dati. */
    if (thomas_factor(m, low_est, diag_est, up_est, beta, alpha) != 0) {
        fprintf(stderr, "rank %d: pivot nullo nella fattorizzazione di A_i\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ||b||_2 globale, contando solo le righe mie */
    double nb_loc = 0.0;
    for (int i = 0; i < n_loc; i++) nb_loc += b[i] * b[i];
    const double nb = norma_globale(nb_loc);

    /* a inizio giro k in ue c'è u^{(k)} */
    int k;
    for (k = 0; ; k++) {

        /* 1) residuo sulle righe estese: r_loc = R_i (b - A u) */
        bordi_scambia(ue, n_loc, larg, sx, dx);
        for (int j = 0; j < m; j++) {
            const int i = off + j;
            r_loc[j] = b_est[j] - low_est[j] * ue[i - 1] - diag_est[j] * ue[i]
                       - up_est[j] * ue[i + 1];
        }

        /* nella norma metto solo le righe mie, le altre le conta il vicino */
        double nr_loc = 0.0;
        for (int j = dl; j < dl + n_loc; j++)
            nr_loc += r_loc[j] * r_loc[j];

        /* controllo se mi posso fermare, con il residuo appena calcolato */
        if (norma_globale(nr_loc) / nb <= tol) break;   /* la soluzione è u^{(k)} */

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

    free(low_a); free(diag_a); free(up_a); free(b_a); free(ue);
    free(r_loc); free(w);
    free(alpha); free(beta); free(y);
    return k;
}
