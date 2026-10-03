/*
 * thomas.c — algoritmo di Thomas per sistemi tridiagonali.
 *
 * Fattorizzazione LU specializzata (senza pivotazione) seguita da
 * sostituzione in avanti e all'indietro; costo O(n).
 */

#include <math.h>
#include "solver.h"

int thomas_solve(int n, const double *low, const double *diag,
                 const double *up, const double *f, double *sol,
                 double *alpha, double *y)
{
    /* Fattorizzazione: 
     * alpha_1 = d_1,
     * beta_i = l_i / alpha_{i-1}, 
     * alpha_i = d_i - beta_i s_{i-1}.
     * beta_i non viene memorizzato: serve solo per aggiornare y. */
    alpha[0] = diag[0];
    if (alpha[0] == 0.0) return -1;
    y[0] = f[0];

    for (int i = 1; i < n; i++) {
        double beta = low[i] / alpha[i - 1];
        alpha[i] = diag[i] - beta * up[i - 1];
        if (alpha[i] == 0.0) return -1;
        y[i] = f[i] - beta * y[i - 1];   /* L y = f */
    }

    /* Sostituzione all'indietro: U sol = y */
    sol[n - 1] = y[n - 1] / alpha[n - 1];
    for (int i = n - 2; i >= 0; i--)
        sol[i] = (y[i] - up[i] * sol[i + 1]) / alpha[i];

    return 0;
}