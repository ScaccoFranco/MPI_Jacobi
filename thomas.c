/*
 * thomas.c
 *
 * Algoritmo di Thomas per i sistemi tridiagonali: e' la fattorizzazione LU
 * adattata al caso tridiagonale (senza pivotazione), seguita dalla
 * sostituzione in avanti e da quella all'indietro. Costa O(n).
 * Ho tenuto separate la fattorizzazione (3(n-1) flops) e le sostituzioni
 * (5n-4 flops), perche' nello Schwarz il blocco locale e' sempre lo stesso a
 * ogni iterazione e quindi basta fattorizzarlo una volta sola.
 */

#include "solver.h"

int thomas_factor(int n, const double *low, const double *diag,
                  const double *up, double *beta, double *alpha)
{
    /* Fattorizzazione:
     * alpha_1 = d_1,
     * beta_i = l_i / alpha_{i-1},
     * alpha_i = d_i - beta_i s_{i-1}. */
    alpha[0] = diag[0];
    if (alpha[0] == 0.0) return -1;
    beta[0] = 0.0;                        /* non usato */

    for (int i = 1; i < n; i++) {
        beta[i] = low[i] / alpha[i - 1];
        alpha[i] = diag[i] - beta[i] * up[i - 1];
        if (alpha[i] == 0.0) return -1;
    }
    return 0;
}

void thomas_solve_factored(int n, const double *beta, const double *alpha,
                           const double *up, const double *f, double *sol,
                           double *y)
{
    /* L y = f */
    y[0] = f[0];
    for (int i = 1; i < n; i++)
        y[i] = f[i] - beta[i] * y[i - 1];

    /* Sostituzione all'indietro: U sol = y */
    sol[n - 1] = y[n - 1] / alpha[n - 1];
    for (int i = n - 2; i >= 0; i--)
        sol[i] = (y[i] - up[i] * sol[i + 1]) / alpha[i];
}
