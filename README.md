# MPI_Jacobi

Risoluzione parallela dell'equazione di Poisson 1D

$$-u'' = 1 \text{ su } (0,1), \qquad u(0) = u(1) = 0$$

con il metodo di Jacobi e con il metodo di Schwarz additivo (variante RAS, con
sovrapposizione di `delta` nodi per lato e l'algoritmo di Thomas come solutore
locale), in C con MPI. Con `delta = 0` lo Schwarz coincide con il Jacobi a blocchi.

Il problema è discretizzato alle differenze finite su `N` nodi interni, con
`h = 1/(N+1)` e `A = h^{-2} tridiag(-1, 2, -1)`. La soluzione esatta è
`u(x) = x(1-x)/2`.

## Requisiti

- compilatore C (C11)
- un'implementazione di MPI (Open MPI o MPICH)
- CMake ≥ 3.20

## Compilazione

```bash
cmake -S . -B build
cmake --build build
```

Senza indicazioni si compila in Release (`-O3 -DNDEBUG`). Per una build di debug:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

L'eseguibile è `build/solver`.

## Esecuzione

```bash
mpirun -np <p> ./build/solver <N> <jacobi|schwarz> [tol] [max_iter] [delta]
```

| argomento  | significato                                   | default    |
|------------|-----------------------------------------------|------------|
| `p`        | numero di processi MPI                        | —          |
| `N`        | numero di nodi interni, multiplo di `p`       | —          |
| metodo     | `jacobi` oppure `schwarz`                     | —          |
| `tol`      | tolleranza sul residuo relativo `‖b − Ax‖₂ / ‖b‖₂` | `1e-6` |
| `max_iter` | numero massimo di iterazioni                  | `10000000` |
| `delta`    | solo `schwarz`: nodi di sovrapposizione per lato, con `0 ≤ delta ≤ N/p − 1` | `0` |

### Esempi

```bash
# Jacobi, 128 nodi interni, 4 processi
mpirun -np 4 ./build/solver 128 jacobi

# Schwarz, 1024 nodi interni, 2 processi, tolleranza 1e-8
mpirun -np 2 ./build/solver 1024 schwarz 1e-8

# Schwarz con sovrapposizione di 16 nodi per lato
mpirun -np 2 ./build/solver 2048 schwarz 1e-6 10000000 16

# numero di iterazioni fisso (tol = 0): 1000 iterazioni, utile per misurare i tempi
mpirun -np 4 ./build/solver 1000000 schwarz 0 1000

# processi fissati ai core (Open MPI), per misure più stabili
mpirun -np 4 --bind-to core ./build/solver 16384 jacobi 0 20000
```

### Output

Il rank 0 stampa una riga (per Schwarz anche `delta`):

```
metodo=schwarz  N=1024    p=2   iterazioni=8325     tempo=0.1503 s  err_max=1.102e-08  delta=0
```

- `iterazioni`: iterazioni eseguite (`max_iter + 1` se la tolleranza non è stata raggiunta, e in quel caso compare anche un avviso);
- `tempo`: tempo del solo solutore, del processo più lento (`MPI_Wtime`);
- `err_max`: errore in norma del massimo rispetto alla soluzione esatta.

Con `tol = 0` la tolleranza non viene mai raggiunta: è normale che compaiano
`iterazioni=max_iter+1` e l'avviso.

Con `f = 1` lo schema alle differenze finite è esatto nei nodi, quindi `err_max`
misura solo l'errore del solutore iterativo.

## File

| file        | contenuto |
|-------------|-----------|
| `main.c`    | lettura degli argomenti, generazione del problema, chiamata del solutore, stampa |
| `poisson.c` | ogni processo genera le sue righe di `A` (tre diagonali) e di `b`; soluzione esatta ed errore |
| `jacobi.c`  | metodo di Jacobi |
| `schwarz.c` | metodo di Schwarz additivo RAS con sovrapposizione `delta` (`delta = 0`: Jacobi a blocchi) |
| `thomas.c`  | algoritmo di Thomas: fattorizzazione e sostituzioni separate |
| `bordi.c`   | scambio delle celle di overlap con i processi vicini (`MPI_Sendrecv`): 1 valore per lato per Jacobi, `delta + 1` per Schwarz |
| `solver.h`  | interfacce comuni |
