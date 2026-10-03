/*
 * jacobi.c
 *
 * Metodo di Jacobi parallelizzato con MPI.
 *
 * L'idea di fondo e' che ogni processo si tiene solo le sue righe della matrice (n_loc righe a testa), 
 * ma per aggiornare le proprie incognite ha comunque bisogno di TUTTO il vettore soluzione dell'iterazione precedente. 
 * Per questo dopo ogni aggiornamento rimetto insieme il vettore globale con una Allgather.
 *
 * La formula e' quella classica di Jacobi
 *
 *     x^{k+1} = x^k + D^{-1} (b - A x^k)
 *
 * che componente per componente diventa
 *
 *     x_g = ( b_g - somma_{j != g} a_gj x_j ) / a_gg .
 */

#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "solver.h"

int MPI_Jacobi(const double *A_local, const double *b_local,
               double *x, int N, int n_loc, int rank,
               double tol, int max_iter)
{
    /* salvo le nuove incognite di questo processo prima di
     * spedirle a tutti. in modo da non modificare i valori della iterazione corrente */
    double *x_new = malloc(n_loc * sizeof(double));

    /* Mi calcolo la norma di b, che mi serve al
     * denominatore del criterio d'arresto (residuo relativo). E' un
     * numero globale, quindi sommo i pezzi locali con una Allreduce. */
    double norm_b_loc = 0.0;
    for (int i = 0; i < n_loc; i++)
        norm_b_loc += b_local[i] * b_local[i];

    double norm_b;
    MPI_Allreduce(&norm_b_loc, &norm_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    norm_b = sqrt(norm_b);

    int k;
    for (k = 1; k <= max_iter; k++) {
        for (int i = 0; i < n_loc; i++) {
            /* i è indice locale, g è l'indice globale */
            int g = rank * n_loc + i;

            const double *riga = &A_local[i * N];

            double somma = 0.0;
            for (int j = 0; j < N; j++) {
                if (j == g) continue;      /* salto il termine diagonale */
                somma += riga[j] * x[j];
            }

            x_new[i] = (b_local[i] - somma) / riga[g];
        }

        /* rimetto insieme il vettore globale: */
        /* Allgather così tutti hanno il vettore completo per l'iterazione dopo */
        MPI_Allgather(x_new, n_loc, MPI_DOUBLE, x, n_loc, MPI_DOUBLE, MPI_COMM_WORLD);

        /* controllo tolleranza per fermarmi */
        /* Calcolo il residuo r = b - A x sulle mie righe e ne accumulo
         * la norma al quadrato; poi sommo tra tutti i processi. */
        double norm_r_loc = 0.0;
        for (int i = 0; i < n_loc; i++) {
            const double *riga = &A_local[i * N];
            double r = b_local[i];
            for (int j = 0; j < N; j++)
                r -= riga[j] * x[j];
            norm_r_loc += r * r;
        }

        double norm_r;
        MPI_Allreduce(&norm_r_loc, &norm_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* Tutti i processi hanno lo stesso norm_r, quindi decidono tutti insieme di uscire: nessuno resta "indietro" nel ciclo. */
        if (sqrt(norm_r) / norm_b <= tol)
            break;
    }

    free(x_new);

    /* Returno x (globale, uguale su tutti) tramite il puntatore, e il numero di iterazioni fatte come valore di ritorno. */
    return k;
}