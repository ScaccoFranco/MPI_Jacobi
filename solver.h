/*
    solver.h — interfacce comuni del progetto

    La matrice non e' piena: ogni processo tiene solo le sue n_loc righe
    come tre diagonali low, diag, up (lunghezza n_loc), e durante le iterazioni
    il vettore globale non esiste: ogni processo tiene u[0 .. n_loc+1], dove
    u[1..n_loc] sono le sue incognite e u[0], u[n_loc+1] sono celle di overlap
    con i valori dei vicini.

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

/* Ogni processo costruisce le proprie n_loc righe di
   A = h^{-2} tridiag(-1, 2, -1) e del termine noto (f == 1), h = 1/(N+1).
   low[i] e up[i] sono i coefficienti delle colonne g-1 e g+1: valgono 0
   solo nella prima e nell'ultima riga globale. */
void poisson_generate_local(int N, int n_loc, int rank,
                            double *low, double *diag, double *up, double *b);

/*  Soluzione esatta u(x) = x(1-x)/2 valutata nel nodo globale g
    (g = 0 corrisponde a x_1 = h). */
double poisson_exact(int N, int g);

/* Errore in norma del massimo sulle incognite locali u_loc[0..n_loc-1]. */
double poisson_max_error_local(int N, int n_loc, int rank, const double *u_loc);




/*  thomas.c: solutore diretto tridiagonale  */

/* Il sistema tridiagonale T sol = f ha sottodiagonale low[1..n-1],
 * diagonale diag[0..n-1], sopradiagonale up[0..n-2].
 * Nessuna pivotazione: adeguato per matrici a dominanza diagonale o SPD. */

/* Solo fattorizzazione: pivot alpha[0..n-1] e moltiplicatori beta[1..n-1].
   Ritorna 0 in caso di successo, -1 se incontra un pivot nullo. */
int thomas_factor(int n, const double *low, const double *diag,
                  const double *up, double *beta, double *alpha);

/* Solo sostituzioni L y = f e U sol = y, con i fattori gia' calcolati.
   y e' un vettore di lavoro di dimensione n. */
void thomas_solve_factored(int n, const double *beta, const double *alpha,
                           const double *up, const double *f, double *sol,
                           double *y);




/*  bordi.c: scambio delle celle di overlap con i vicini  */

/* Al ritorno u[0] e u[n_loc+1] contengono i valori dei vicini.
   Agli estremi (sx o dx == MPI_PROC_NULL) la cella di overlap resta invariata. */
void bordi_scambia(double *u, int n_loc, int sx, int dx);




/*  jacobi.c / schwarz.c: solutori iterativi MPI  */

/* Entrambi risolvono A x = b con il criterio d'arresto
 *   ||b - A x||_2 / ||b||_2 <= tol
 * e ritornano il numero di iterazioni eseguite (oppure max_iter+1 se la tolleranza non è stata raggiunta).
 * u ha dimensione n_loc + 2 e all'ingresso contiene l'innesco (celle di overlap
 * comprese, che devono valere 0); al ritorno u[1..n_loc] contiene la soluzione locale.
*/
int MPI_Jacobi (const double *low, const double *diag, const double *up,
                const double *b, double *u, int n_loc, int rank, int size,
                double tol, int max_iter);

int MPI_Schwarz(const double *low, const double *diag, const double *up,
                const double *b, double *u, int n_loc, int rank, int size,
                double tol, int max_iter);

#endif
