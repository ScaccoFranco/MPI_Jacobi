# Implementazione del metodo di Jacobi con MPI

## Struttura del progetto

Il progetto è organizzato in tre file:

- `main.c` — punto di ingresso, inizializzazione MPI, distribuzione dati, stampa risultati
- `jacobi.c` — implementazione del metodo di Jacobi (seriale e parallela) e generazione matrici
- `jacobi.h` — dichiarazioni delle funzioni

---

## Generazione delle matrici

Per testare il metodo ho bisogno di un sistema lineare `Ax = b` con soluzione nota, in modo da poter verificare la correttezza dell'implementazione. Ho implementato tre funzioni di generazione in `jacobi.c`.

### Generatore di numeri pseudo-casuali (LCG)

Per avere risultati riproducibili tra un'esecuzione e l'altra ho implementato un generatore lineare congruenziale (LCG) invece di usare `rand()` della libreria standard, che dipende dallo stato globale del processo e non garantisce riproducibilità in ambiente MPI.

La formula ricorsiva è:

```
X(n+1) = (a * X(n) + c) mod m
```

Con i parametri:
- `a = 1664525` (moltiplicatore di Knuth)
- `c = 1013904223` (incremento da Numerical Recipes)
- `m = 2^32` (modulo implicito lavorando con `uint32_t`)

```c
static double rand_double(unsigned int *seed) {
    *seed = (*seed * 1664525u + 1013904223u);
    return ((double)(*seed) / (double)UINT32_MAX) * 2.0 * 2.0 - 2.0;
}
```

### Matrice diagonalmente dominante con soluzione unitaria

La funzione principale che uso negli esperimenti è `genera_matrici_diagdom_xones`. Genera una matrice diagonalmente dominante per righe — condizione sufficiente per garantire la convergenza del metodo di Jacobi — con termine noto `b` costruito in modo che la soluzione esatta sia `x* = [1, 1, ..., 1]`.

Per farlo, calcolo prima gli elementi fuori diagonale casuali, poi impongo:

```
a_ii = somma_{j != i} |a_ij| + margine
b_i  = somma_{j=0}^{n-1} a_ij   (con x* = 1 => b = A * ones)
```

```c
void genera_matrici_diagdom_xones(double *A, double *b, const int n, unsigned int seed) {
    for (int i = 0; i < n; i++) {
        double somma_righe = 0.0;
        b[i] = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                A[i * n + j] = rand_double(&seme);
                somma_righe += fabs(A[i * n + j]);
                b[i] += A[i * n + j];
            }
        }
        A[i * n + i] = somma_righe + margine;
        b[i] += A[i * n + i];
    }
}
```

---

## Implementazione seriale di Jacobi

La versione seriale `jacobi()` implementa direttamente la formula iterativa:

```
x_i^(k+1) = ( b_i - sommatoria_{j != i} A_ij * x_j^(k) ) / A_ii
```

Uso due buffer `x` e `x_new` per separare la lettura (iterata `k`) dalla scrittura (iterata `k+1`), come richiede il metodo di Jacobi (a differenza di Gauss-Seidel che userebbe i valori aggiornati in-place).

Il criterio di arresto è la **norma infinito** dell'incremento tra due iterate:

```
diff = max_i | x_new_i - x_i |
```

---

## Implementazione parallela con MPI

### Strategia di decomposizione

Ho scelto una decomposizione per **blocchi di righe**: ogni processo riceve `size / world_size` righe consecutive della matrice `A` e il corrispondente blocco di `b`. Questa scelta è naturale per il metodo di Jacobi perché ogni componente `x_i^(k+1)` dipende solo dalla riga `i` di `A` e dall'intera soluzione `x^(k)` del passo precedente — quindi il calcolo di ogni blocco di righe è indipendente e può essere parallelizzato senza comunicazione durante il calcolo.

### Distribuzione dei dati nel `main`

Ho scelto deliberatamente di fare la distribuzione dei dati nel `main` e non dentro la funzione `MPI_Jacobi`. Il motivo è ottimizzare l'uso della memoria: in questo modo solo il processo di rank 0 alloca e genera la matrice completa, mentre tutti gli altri allocano solo la propria porzione locale.

```c
double *A_local = malloc(per_proc * size * sizeof(double));
double *b_local = malloc(per_proc * sizeof(double));
double *A = NULL;
double *b = NULL;

if (rank == 0) {
    A = malloc(size * size * sizeof(double));
    b = malloc(size * sizeof(double));
    genera_matrici_diagdom_xones(A, b, size, seed);
}

MPI_Scatter(A, per_proc * size, MPI_DOUBLE, A_local, per_proc * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Scatter(b, per_proc,        MPI_DOUBLE, b_local, per_proc,        MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

`MPI_Scatter` divide automaticamente il buffer del processo root in `world_size` blocchi uguali e ne invia uno a ciascun processo. I processi non-rank-0 passano `A = NULL` come send buffer: questo è corretto perché MPI ignora il send buffer nei processi non-root.

### Iterazione parallela

Ogni processo lavora solo sulle proprie righe locali. L'indice globale di riga si calcola come:

```c
int global_j = row_start + j;   // row_start = rank * per_proc
```

Il calcolo della sommatoria usa gli indici locali per accedere ad `A_local`, ma gli indici globali per accedere a `x` (che ogni processo mantiene completo):

```c
for (int j = 0; j < per_proc; j++) {
    double sommatoria = 0.0;
    int global_j = row_start + j;
    for (int k = 0; k < size; k++) {
        if (k != global_j) {
            sommatoria += A[j * size + k] * x[k];
        }
    }
    x_new[global_j] = (b[j] - sommatoria) / A[j * size + global_j];
}
```

### Comunicazioni collettive

Alla fine di ogni iterazione sono necessarie due comunicazioni collettive.

**`MPI_Allgather`** — raccoglie le porzioni di `x_new` calcolate da ciascun processo e le assembla nel vettore `x` completo su tutti i processi. Questo è necessario perché all'iterazione successiva ogni processo ha bisogno dell'intera soluzione `x^(k)` per calcolare la propria sommatoria.

```c
MPI_Allgather(x_new + row_start, per_proc, MPI_DOUBLE,
              x,                 per_proc, MPI_DOUBLE,
              MPI_COMM_WORLD);
```

Ho scelto `MPI_Allgather` invece di `MPI_Gather` + `MPI_Bcast` perché combina le due operazioni in una sola, riducendo il numero di chiamate MPI.

**`MPI_Allreduce`** — calcola il massimo globale della `local_diff` tra tutti i processi. Serve per verificare il criterio di convergenza in modo coerente: se usassi solo la diff locale, processi diversi potrebbero avere opinioni diverse su quando fermarsi, causando un deadlock.

```c
MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
if (global_diff < toll) break;
```

Anche qui uso `Allreduce` invece di `Reduce` + `Bcast` perché il risultato serve a tutti i processi, non solo al root.

---

## Gestione della memoria

Seguendo il modello SPMD di MPI, tutti i processi eseguono lo stesso codice dall'inizio. Le allocazioni che riguardano solo rank 0 sono quindi protette da `if (rank == 0)` per evitare di sprecare memoria sugli altri processi.

Per le strutture dati locali (`A_local`, `b_local`, `x0`) l'allocazione avviene prima di `MPI_Scatter` su tutti i processi — questo è corretto perché ogni processo ha bisogno del proprio buffer di ricezione prima che la scatter avvenga.

All'interno di `MPI_Jacobi` i buffer `x` e `x_new` vengono allocati con `malloc` e inizializzati con `memcpy` da `x0`:

```c
double *x     = malloc(size * sizeof(double));
double *x_new = malloc(size * sizeof(double));
memcpy(x, x0, size * sizeof(double));
```

La funzione restituisce il puntatore `x` (la soluzione finale) e libera `x_new`. La liberazione di `x` è responsabilità del chiamante (il `main`).

---

## Compilazione ed esecuzione

```bash
# Compilazione
mpicc main.c jacobi.c -o mpi_jacobi -lm

# Esecuzione seriale (1 processo)
mpirun -np 1 mpi_jacobi

# Esecuzione parallela con N processi
mpirun -np N mpi_jacobi
```

Il flag `-lm` è necessario per linkare la libreria matematica (`fabs`, `fmax` da `<math.h>`).