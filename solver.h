/*
    solver.h — interfacce comuni del progetto

    Variabili:
    N         numero di incognite globali (multiplo di size)
    n_loc     numero di righe/incognite locali = N / size
    rank      identificativo del processo, 0 <= rank < size
    A_local   blocco di n_loc righe consecutive della matrice, memorizzato per righe (row-major), dimensione n_loc x N
    b_local   porzione locale del termine noto, dimensione n_loc
    x         vettore soluzione GLOBALE, dimensione N, replicato su ogni processo e riassemblato a ogni iterazione con MPI_Allgather
  
    La riga locale i corrisponde alla riga globale:
        g = rank * n_loc + i
 */

#ifndef SOLVER_H
#define SOLVER_H

/* poisson.c: generazione del problema  */

// TODO: ora uso una f = 1, provare poi a generalizzare o cmq provare altre f

/* Assembla (sul chiamante) la matrice densa N x N del problema di
   Poisson 1D, A = h^{-2} tridiag(-1, 2, -1), e il termine noto per
   f == 1 con dati al bordo nulli. h = 1/(N+1). */
void poisson_generate(int N, double *A, double *b);

/*  Soluzione esatta u(x) = x(1-x)/2 valutata nel nodo globale g
    (g = 0 corrisponde a x_1 = h). */
double poisson_exact(int N, int g);

/* Errore in norma del massimo rispetto alla soluzione esatta. */
double poisson_max_error(int N, const double *x);




/*  thomas.c: solutore diretto tridiagonale  */

/* Risolve il sistema tridiagonale T sol = rhs, dove T ha
 * sottodiagonale low[1..n-1], diagonale diag[0..n-1], sopradiagonale up[0..n-2]. 
 * Usa i buffer di lavoro beta e y (dimensione n) forniti dal chiamante. 
 * Nessuna pivotazione:
 * adeguato per matrici a dominanza diagonale o SPD.
 * Ritorna 0 in caso di successo, -1 se incontra un pivot nullo. */
int thomas_solve(int n, const double *low, const double *diag,
                 const double *up, const double *rhs, double *sol,
                 double *beta, double *y);



/*  jacobi.c / schwarz.c: solutori iterativi MPI  */

/* Entrambi risolvono A x = b con il criterio d'arresto
 *   ||b - A x||_2 / ||b||_2 <= tol
 * e ritornano il numero di iterazioni eseguite (oppure max_iter+1 se la tolleranza non è stata raggiunta). 
 * Al ritorno x contiene la soluzione globale, identica su tutti i processi. 
*/
int MPI_Jacobi (const double *A_local, const double *b_local,
                double *x, int N, int n_loc, int rank,
                double tol, int max_iter);

int MPI_Schwarz(const double *A_local, const double *b_local,
                double *x, int N, int n_loc, int rank,
                double tol, int max_iter);

#endif /* SOLVER_H */