/*
 * bordi.c
 *
 * Qui scambio le celle di overlap con i processi vicini.
 *
 * Ogni processo tiene u[0 .. n_loc + 2*larg - 1]. Le sue incognite sono in
 * u[larg .. larg+n_loc-1], in u[0 .. larg-1] ci metto le ultime larg incognite
 * del vicino di sinistra e in u[larg+n_loc .. n_loc+2*larg-1] le prime larg
 * del vicino di destra.
 *
 * Jacobi usa larg = 1: A e' tridiagonale, quindi per fare A u sulle mie righe
 * mi basta un valore per lato. Schwarz con sovrapposizione delta usa
 * larg = delta + 1, perche' il residuo lo calcola anche sui delta nodi presi
 * dal vicino, e per la loro riga serve un valore in piu'.
 *
 * Agli estremi il vicino e' MPI_PROC_NULL: la send non fa niente e la receive
 * non tocca il buffer. Cosi' le celle di overlap del rank 0 a sinistra e
 * dell'ultimo rank a destra restano a 0, che sono proprio le condizioni al
 * bordo u(0) = u(1) = 0.
 * Uso MPI_Sendrecv cosi' non devo mettere in ordine a mano send e receive
 * per evitare il deadlock.
 */

#include <mpi.h>
#include "solver.h"

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
