#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "jacobi.h"

int main (int argc, char** argv) {

    // faccio sempre nel main, non nella funzione!!
    MPI_Init(NULL, NULL);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    
    int size = 1000;
    int per_proc = size / world_size; //Righe assegnate a un singolo processo
    unsigned int seed = 1000;
    double toll = 1e-3;
    unsigned int itermax = 1000U;

    // Variabili dei singoli processi:
    double *A_local = malloc(per_proc * size * sizeof(double)); 
    double *b_local = malloc(per_proc * sizeof(double)); 
    double *x0 = malloc(size * sizeof(double));
    double *A = NULL;
    double *b = NULL;

    // Definisco le matrici per intere solo in un processo così da ottimizzare l'utiizzo della memoria
    // altrienti avrei N*memoria occupata, invece che solo 2*memoria occupata:
    // memoria matrice nel processso 0 + n-esime parti della memoria negli altri processi
    if (rank == 0) {
        A = malloc(size * size * sizeof(double)); 
        b = malloc(size * sizeof(double)); 
        genera_matrici_diagdom_xones(A, b, size, seed);
    }

    // Ora devo distribuire A e b ai vari processi:
    MPI_Scatter(A, per_proc * size, MPI_DOUBLE, A_local, per_proc * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, per_proc, MPI_DOUBLE, b_local, per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    fill_const(x0, 0.0, size, 1); // tipo zeros di matlab

    // double *sol = jacobi(A, b, x0, size, toll, itermax);
    double *sol = MPI_Jacobi(A, b, x0, size, toll, itermax);


    if (rank == 0) {
        for (int i = 0; i < size; i++) { // stampo x finale
            printf("%.4f\n", sol[i]);
        }

        free(A);
        free(b);
    }
    

    free(A_local);
    free(b_local);
    free(x0);
    free(sol);
    //pure questo main e non funzione
    MPI_Finalize();

    return 0;
}