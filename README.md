## Istruzioni:
Per compilare: ( aggiunto link libreria matematica)
```
mpicc main.c jacobi.c -o mpi_jacobi -lm
```

per runnare:
```
mpirun mpi_jacobi
```
però se vogglio usare N processi scrivo
```
mpirun -np N mpi_jacobi
```
