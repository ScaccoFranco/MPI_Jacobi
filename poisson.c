/*
 * poisson.c — problema modello: -u'' = 1 su (0,1), u(0) = u(1) = 0.
 *
 * Discretizzazione alle differenze finite centrate su N nodi interni
 * x_{g+1} = (g+1) h, g = 0..N-1, con h = 1/(N+1):
 *
 *     A = h^{-2} tridiag(-1, 2, -1),   b_g = f(x_{g+1}) = 1.
 *
 * Non si assembla la matrice piena: ogni processo genera solo le tre
 * diagonali delle proprie righe.
 *
 * Soluzione esatta: u(x) = x (1 - x) / 2.
 */

#include <math.h>
#include "solver.h"

void poisson_generate_local(int N, int n_loc, int rank,
                            double *low, double *diag, double *up, double *b)
{
    const double h = 1.0 / (double)(N + 1);
    const double invh2 = 1.0 / (h * h);

    for (int i = 0; i < n_loc; i++) {
        int g = rank * n_loc + i;      /* riga globale */
        diag[i] = 2.0 * invh2;
        low[i] = (g > 0) ? -invh2 : 0.0;
        up[i] = (g < N - 1) ? -invh2 : 0.0;
        b[i] = 1.0;                    /* f(x) = 1 */
    }
}

double poisson_exact(int N, int g)
{
    const double h = 1.0 / (double)(N + 1);
    const double xg = (double)(g + 1) * h;
    return 0.5 * xg * (1.0 - xg);
}

double poisson_max_error_local(int N, int n_loc, int rank, const double *u_loc)
{
    double err = 0.0;
    for (int i = 0; i < n_loc; i++) {
        double e = fabs(u_loc[i] - poisson_exact(N, rank * n_loc + i));
        if (e > err) err = e;
    }
    return err;
}
