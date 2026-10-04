/*
 * jacobi.c
 *
 * Metodo di Jacobi parallelizzato con MPI.
 *
 * Ogni processo si tiene solo le sue righe della matrice (n_loc righe a testa).
 * La formula e' quella classica di Jacobi
 *
 *     x^{k+1} = x^k + D^{-1} (b - A x^k)
 *
 * che componente per componente diventa
 *
 *     x_g = ( b_g - somma_{j != g} a_gj x_j ) / a_gg .
 *
 * A e' tridiagonale, quindi la somma ha solo due termini, e ogni processo
 * ha bisogno solo dei due valori di bordo dei vicini (celle di overlap) invece
 * di tutto il vettore: al posto di MPI_Allgather bastano due MPI_Sendrecv.
 *
 * Residuo senza calcoli in piu': nello stesso passaggio in cui calcolo x^{k+1}
 * ottengo anche r^{(k)} = b - A x^{(k)}, perche' t = b - somma e' gia' la parte
 * comune: x^{k+1}_g = t / a_gg e r^{(k)}_g = t - a_gg x^{(k)}_g.
 * Il residuo che controllo e' quindi quello dell'iterata PRECEDENTE: se e' gia'
 * sotto tolleranza butto via x^{k+1} e restituisco x^{(k)}.
 */

#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "solver.h"

int MPI_Jacobi(const double *low, const double *diag, const double *up,
               const double *b, double *u, int n_loc, int rank, int size,
               double tol, int max_iter)
{
    /* vicini a sinistra e a destra (MPI_PROC_NULL agli estremi) */
    const int sx = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    const int dx = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Mi serve un secondo vettore, anche lui con le celle di overlap: a fine
     * giro scambio i puntatori invece di copiare. Uso calloc cosi' le celle ai
     * bordi fisici restano 0. */
    double *cur = u;
    double *nxt = calloc((size_t)n_loc + 2, sizeof(double));

    /* Mi calcolo la norma di b, che mi serve al
     * denominatore del criterio d'arresto (residuo relativo). E' un
     * numero globale, quindi sommo i pezzi locali con una Allreduce. */
    double norm_b_loc = 0.0;
    for (int i = 0; i < n_loc; i++)
        norm_b_loc += b[i] * b[i];
    const double norm_b = norma_globale(norm_b_loc);

    /* All'inizio del giro k, cur contiene x^{(k)}. */
    int k;
    for (k = 0; ; k++) {
        bordi_scambia(cur, n_loc, 1, sx, dx);

        double norm_r_loc = 0.0;
        for (int i = 1; i <= n_loc; i++) {
            double somma = low[i - 1] * cur[i - 1] + up[i - 1] * cur[i + 1];
            double t = b[i - 1] - somma;
            double r = t - diag[i - 1] * cur[i];   /* r^{(k)} sulla riga i */
            nxt[i] = t / diag[i - 1];              /* x^{(k+1)} */
            norm_r_loc += r * r;
        }

        /* controllo tolleranza per fermarmi. La norma e' uguale per tutti i
         * processi, quindi escono tutti insieme e nessuno resta "indietro"
         * nel ciclo. */
        if (norma_globale(norm_r_loc) / norm_b <= tol)
            break;                       /* soluzione: x^{(k)} in cur */
        if (k == max_iter) { k = max_iter + 1; break; }

        double *tmp = cur; cur = nxt; nxt = tmp;
    }

    /* rimetto la soluzione nel vettore di chi mi ha chiamato */
    if (cur != u) {
        memcpy(u, cur, ((size_t)n_loc + 2) * sizeof(double));
        free(cur);
    } else {
        free(nxt);
    }
    return k;
}
