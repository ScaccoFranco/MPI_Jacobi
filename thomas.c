/*
 * thomas.c — algoritmo di Thomas per sistemi tridiagonali.
 *
 * Fattorizzazione LU specializzata (senza pivotazione) seguita da
 * sostituzione in avanti e all'indietro; costo O(n).
 *
 * Riferimento: Quarteroni et al., "Matematica Numerica", 4a ed.,
 * Springer 2014, Sez. 3.7.1.
 */

#include <math.h>
#include "solver.h"

int thomas_solve(int n, const double *low, const double *diag,
                 const double *up, const double *rhs, double *sol,
                 double *beta, double *y)
{
    /* Fattorizzazione: 
     * beta_1 = d_1,
     * gamma_i = l_i / beta_{i-1}, 
     * beta_i = d_i - gamma_i s_{i-1}.
     * gamma_i non viene memorizzato: serve solo per aggiornare y. */
    beta[0] = diag[0];
    if (beta[0] == 0.0) return -1;
    y[0] = rhs[0];

    for (int i = 1; i < n; i++) {
        double gamma = low[i] / beta[i - 1];
        beta[i] = diag[i] - gamma * up[i - 1];
        if (beta[i] == 0.0) return -1;
        y[i] = rhs[i] - gamma * y[i - 1];   /* L y = rhs */
    }

    /* Sostituzione all'indietro: U sol = y */
    sol[n - 1] = y[n - 1] / beta[n - 1];
    for (int i = n - 2; i >= 0; i--)
        sol[i] = (y[i] - up[i] * sol[i + 1]) / beta[i];

    return 0;
}