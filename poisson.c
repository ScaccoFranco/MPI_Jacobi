/*
 * poisson.c — problema modello: -u'' = 1 su (0,1), u(0) = u(1) = 0.
 *
 * Discretizzazione alle differenze finite centrate su N nodi interni
 * x_{g+1} = (g+1) h, g = 0..N-1, con h = 1/(N+1):
 *
 *     A = h^{-2} tridiag(-1, 2, -1),   b_g = f(x_{g+1}) = 1.
 *
 * Soluzione esatta: u(x) = x (1 - x) / 2.
 *
 * Riferimenti: Quarteroni, Sacco, Saleri, Gervasio, "Matematica Numerica", 4a ed., Springer 2014, Sez. 11.2.
 */

#include <math.h>
#include <string.h>
#include "solver.h"

void poisson_generate(int N, double *A, double *b)
{
    const double h = 1.0 / (double)(N + 1);
    const double invh2 = 1.0 / (h * h);

    memset(A, 0, (size_t)N * (size_t)N * sizeof(double));
    for (int g = 0; g < N; g++) {
        A[(size_t)g * N + g] = 2.0 * invh2;
        if (g > 0) A[(size_t)g * N + (g - 1)] = -invh2;
        if (g < N - 1) A[(size_t)g * N + (g + 1)] = -invh2;
        b[g] = 1.0;                    /* f(x) = 1 */
    }
}

double poisson_exact(int N, int g)
{
    const double h = 1.0 / (double)(N + 1);
    const double xg = (double)(g + 1) * h;
    return 0.5 * xg * (1.0 - xg);
}

double poisson_max_error(int N, const double *x)
{
    double err = 0.0;
    for (int g = 0; g < N; g++) {
        double e = fabs(x[g] - poisson_exact(N, g));
        if (e > err) err = e;
    }
    return err;
}