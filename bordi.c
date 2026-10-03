/*
 * bordi.c — scambio delle celle di overlap tra processi vicini.
 *
 * Ogni processo tiene u[0 .. n_loc+1]: u[1..n_loc] sono le sue incognite,
 * u[0] e' l'ultima incognita del vicino di sinistra e u[n_loc+1] la prima
 * del vicino di destra. Per il prodotto A u sulle righe locali serve solo questo,
 * perche' A e' tridiagonale.
 *
 * Agli estremi il vicino e' MPI_PROC_NULL: la send non ha effetto e la receive
 * non modifica il buffer, quindi u[0] del rank 0 e u[n_loc+1] dell'ultimo rank
 * restano a 0, cioe' le condizioni al bordo u(0) = u(1) = 0.
 * MPI_Sendrecv evita di dover ordinare a mano send e receive per non andare
 * in deadlock.
 */

#include <mpi.h>
#include "solver.h"

void bordi_scambia(double *u, int n_loc, int sx, int dx)
{
    /* mando il mio primo valore a sinistra, ricevo la cella di overlap di destra */
    MPI_Sendrecv(&u[1],         1, MPI_DOUBLE, sx, 0,
                 &u[n_loc + 1], 1, MPI_DOUBLE, dx, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* mando il mio ultimo valore a destra, ricevo la cella di overlap di sinistra */
    MPI_Sendrecv(&u[n_loc],     1, MPI_DOUBLE, dx, 1,
                 &u[0],         1, MPI_DOUBLE, sx, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
