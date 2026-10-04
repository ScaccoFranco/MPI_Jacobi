/*
    solver.h

    Qui ci sono le funzioni che usano i vari file.

    La matrice piena non c'e': ogni processo tiene solo le sue n_loc righe,
    salvate come tre diagonali low, diag, up (lunghe n_loc). Anche il vettore
    globale durante le iterazioni non esiste: ogni processo tiene
    u[0 .. n_loc+1], dove u[1..n_loc] sono le sue incognite e u[0], u[n_loc+1]
    sono le celle di overlap con i valori dei vicini.

    Variabili:
    N         numero di incognite globali (multiplo di size)
    n_loc     numero di righe/incognite locali = N / size
    rank      identificativo del processo, 0 <= rank < size

    La riga locale i (0-based nei vettori low/diag/up/b) corrisponde alla
    riga globale g = rank * n_loc + i e all'incognita u[i + 1].
 */

#ifndef SOLVER_H
#define SOLVER_H

/* poisson.c: generazione del problema  */

// TODO: ora uso una f = 1, provare poi a generalizzare o cmq provare altre f

/* Ogni processo si costruisce le sue n_loc righe di
   A = h^{-2} tridiag(-1, 2, -1) e del termine noto (f == 1), h = 1/(N+1).
   low[i] e up[i] sono i coefficienti delle colonne g-1 e g+1, e sono 0
   solo nella prima e nell'ultima riga globale. */
void poisson_generate_local(int N, int n_loc, int rank,
                            double *low, double *diag, double *up, double *b);

/*  Soluzione esatta u(x) = x(1-x)/2 valutata nel nodo globale g
    (g = 0 corrisponde a x_1 = h). */
double poisson_exact(int N, int g);

/* Errore in norma del massimo, solo sulle mie incognite u_loc[0..n_loc-1]. */
double poisson_max_error_local(int N, int n_loc, int rank, const double *u_loc);




/*  thomas.c: solutore diretto tridiagonale  */

/* Il sistema tridiagonale T sol = f ha sottodiagonale low[1..n-1],
 * diagonale diag[0..n-1] e sopradiagonale up[0..n-2].
 * Non faccio pivotazione, quindi va bene per matrici a dominanza diagonale
 * o SPD. */

/* Fa solo la fattorizzazione: pivot alpha[0..n-1] e moltiplicatori
   beta[1..n-1]. Ritorna 0 se va tutto bene, -1 se trova un pivot nullo. */
int thomas_factor(int n, const double *low, const double *diag,
                  const double *up, double *beta, double *alpha);

/* Fa solo le sostituzioni L y = f e U sol = y, con i fattori gia' calcolati.
   y e' un vettore d'appoggio lungo n. */
void thomas_solve_factored(int n, const double *beta, const double *alpha,
                           const double *up, const double *f, double *sol,
                           double *y);




/*  schwarz.c: scambio delle celle di overlap e norma globale (le usa anche jacobi.c)  */

/* u ha larg celle di overlap per lato e le mie incognite sono in
   u[larg .. larg+n_loc-1]. Dopo la chiamata in u[0 .. larg-1] e
   u[larg+n_loc .. n_loc+2*larg-1] ci sono i valori dei vicini.
   Agli estremi (sx o dx == MPI_PROC_NULL) le celle di overlap non cambiano.
   Serve larg <= n_loc. */
void bordi_scambia(double *u, int n_loc, int larg, int sx, int dx);

/* Norma 2 globale: riceve la somma dei quadrati delle componenti locali e
   ritorna la radice della somma su tutti i processi (MPI_Allreduce), uguale
   per tutti. */
double norma_globale(double somma_quadrati_loc);




/*  jacobi.c / schwarz.c: solutori iterativi MPI  */

/* Tutti e due risolvono A x = b con il criterio d'arresto
 *   ||b - A x||_2 / ||b||_2 <= tol
 * e ritornano quante iterazioni hanno fatto (oppure max_iter+1 se non hanno
 * raggiunto la tolleranza).
 * u e' lungo n_loc + 2: all'inizio contiene l'innesco (anche le celle di
 * overlap, che devono essere 0), alla fine in u[1..n_loc] c'e' la soluzione
 * locale.
*/
int MPI_Jacobi (const double *low, const double *diag, const double *up,
                const double *b, double *u, int n_loc, int rank, int size,
                double tol, int max_iter);

/* delta e' il numero di nodi di sovrapposizione per lato (0 = Jacobi a
   blocchi), e serve delta + 1 <= n_loc. */
int MPI_Schwarz(const double *low, const double *diag, const double *up,
                const double *b, double *u, int n_loc, int rank, int size,
                double tol, int max_iter, int delta);

#endif
