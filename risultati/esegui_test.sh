#!/bin/bash
# esegui_test.sh
#
# Lancio i test per il capitolo dei risultati e salvo i dati in file CSV
# dentro la cartella dati/, un file per ogni test.
#
# Prima bisogna compilare il programma (dalla cartella del repo):
#   cmake -S . -B build && cmake --build build
#
# Uso:  ./esegui_test.sh                   lancia tutti i test
#       ./esegui_test.sh tempi_processi    lancia solo un test (i nomi sono sotto, nel case)
#
# Per usare un altro eseguibile:  SOLVER=percorso/solver ./esegui_test.sh
#
# ATTENZIONE: per i test sui tempi il portatile deve essere collegato al
# caricatore, altrimenti la CPU va piu' piano e i tempi non sono confrontabili.
#
# Ogni prova la ripeto RIPETIZIONI volte e salvo tutte le ripetizioni (colonna "prova").
# I test correttezza, tolleranza, jacobi_N e schwarz_N_p guardano solo iterazioni
# ed errore, quindi il tempo non lo salvo.

SOLVER=${SOLVER:-../build/solver}
RIPETIZIONI=5      # ogni prova la ripeto 5 volte e salvo tutte le misure; nei grafici faccio la media
mkdir -p dati

# lancia <p> <argomenti del solver>
# lancia il solver con p processi e scrive "iterazioni,tempo,errore"
lancia() {
    p=$1
    shift
    # con piu' processi che core (ne ho 4) serve --oversubscribe
    if [ $p -le 4 ]; then opzioni="--bind-to core"; else opzioni="--oversubscribe"; fi
    riga=$(mpirun $opzioni -np $p $SOLVER "$@" | head -n 1)
    # se il programma non ha stampato la riga dei risultati lo segnalo, invece di salvare dati sbagliati
    if ! echo "$riga" | grep -q "iterazioni="; then
        echo "ERRORE: nessun risultato da: mpirun -np $p $SOLVER $*" >&2
        echo ",,"
        return
    fi
    iterazioni=$(echo "$riga" | sed -E 's/.*iterazioni=([0-9]+).*/\1/')
    tempo=$(echo "$riga" | sed -E 's/.*tempo=([0-9.]+).*/\1/')
    errore=$(echo "$riga" | sed -E 's/.*err_max=([0-9.e+-]+).*/\1/')
    echo "$iterazioni,$tempo,$errore"
}

# lancia_senza_tempo <p> <argomenti del solver>
# come lancia, ma scrive solo "iterazioni,errore" (per i test in cui il tempo non serve)
lancia_senza_tempo() {
    lancia "$@" | cut -d, -f1,3
}

test_scelti=${@:-"correttezza tolleranza jacobi_N schwarz_N_p sovrapposizione tempo_soluzione tempi_processi"}

for t in $test_scelti; do
    echo "test: $t"
    case $t in

    correttezza)
        # stesso problema (N = 1024) con p diversi: Jacobi deve dare sempre lo stesso risultato
        echo "metodo,N,p,prova,iterazioni,errore" > dati/correttezza.csv
        for metodo in jacobi schwarz; do
            for p in 1 2 4; do
                for r in $(seq $RIPETIZIONI); do
                    echo "$metodo,1024,$p,$r,$(lancia_senza_tempo $p 1024 $metodo)" >> dati/correttezza.csv
                done
            done
        done ;;

    tolleranza)
        # come cambiano errore e iterazioni se cambio la tolleranza (N = 1024, p = 4)
        echo "metodo,N,p,tol,prova,iterazioni,errore" > dati/tolleranza.csv
        for metodo in jacobi schwarz; do
            for tol in 1e-4 1e-6 1e-8 1e-10; do
                for r in $(seq $RIPETIZIONI); do
                    echo "$metodo,1024,4,$tol,$r,$(lancia_senza_tempo 4 1024 $metodo $tol)" >> dati/tolleranza.csv
                done
            done
        done ;;

    jacobi_N)
        # iterazioni di Jacobi al variare di N: non dipendono da p, quindi uso p = 1
        echo "N,prova,iterazioni,errore" > dati/jacobi_N.csv
        for N in 32 64 128 256 512 1024; do
            for r in $(seq $RIPETIZIONI); do
                echo "$N,$r,$(lancia_senza_tempo 1 $N jacobi)" >> dati/jacobi_N.csv
            done
        done ;;

    schwarz_N_p)
        # iterazioni di Schwarz senza sovrapposizione al variare di N e di p,
        # con gli stessi N di jacobi_N per poterli confrontare
        echo "N,p,prova,iterazioni,errore" > dati/schwarz_N_p.csv
        for N in 32 64 128 256 512 1024; do
            for p in 2 4 8; do
                for r in $(seq $RIPETIZIONI); do
                    echo "$N,$p,$r,$(lancia_senza_tempo $p $N schwarz)" >> dati/schwarz_N_p.csv
                done
            done
        done ;;

    sovrapposizione)
        # Schwarz con N = 2048 al variare della sovrapposizione delta
        echo "p,delta,prova,iterazioni,tempo,errore" > dati/sovrapposizione.csv
        for p in 2 4; do
            for delta in 0 1 2 4 8 16 32 64 128; do
                for r in $(seq $RIPETIZIONI); do
                    echo "$p,$delta,$r,$(lancia $p 2048 schwarz 1e-6 10000000 $delta)" >> dati/sovrapposizione.csv
                done
            done
        done ;;

    tempo_soluzione)
        # tempo per arrivare alla soluzione con tol = 1e-6, con p = 1, 2, 4.
        # Per Schwarz provo anche una sovrapposizione pari a un quarto del blocco di ogni processo.
        echo "metodo,N,p,delta,prova,iterazioni,tempo,errore" > dati/tempo_soluzione.csv
        for N in 128 256 512 1024; do
            for p in 1 2 4; do
                n_loc=$((N / p))
                delta=$((n_loc / 4))
                for r in $(seq $RIPETIZIONI); do
                    echo "jacobi,$N,$p,0,$r,$(lancia $p $N jacobi)" >> dati/tempo_soluzione.csv
                    echo "schwarz,$N,$p,0,$r,$(lancia $p $N schwarz)" >> dati/tempo_soluzione.csv
                    # con p = 1 c'e' un solo blocco e la sovrapposizione non ha senso
                    if [ $p -gt 1 ]; then
                        echo "schwarz,$N,$p,$delta,$r,$(lancia $p $N schwarz 1e-6 10000000 $delta)" >> dati/tempo_soluzione.csv
                    fi
                done
            done
        done ;;

    tempi_processi)
        # tempo impiegato con p = 1, 2, 4 a numero di iterazioni fisso: con tol = -1 la tolleranza non viene mai raggiunta
        # (con tol = 0 Schwarz con p = 1 si puo' fermare prima, se il residuo diventa esattamente 0).
        # Gli N crescono ogni volta di 8 volte, e scelgo le iterazioni in modo che N * iterazioni sia circa sempre lo stesso.
        echo "metodo,N,p,prova,iterazioni,tempo,errore" > dati/tempi_processi.csv
        for coppia in 2048:250000 16384:30000 131072:4000 1048576:500 8388608:60; do
            N=${coppia%:*}
            it=${coppia#*:}
            for metodo in jacobi schwarz; do
                for p in 1 2 4; do
                    for r in $(seq $RIPETIZIONI); do
                        echo "$metodo,$N,$p,$r,$(lancia $p $N $metodo -1 $it)" >> dati/tempi_processi.csv
                    done
                done
            done
        done ;;

    *)
        echo "test sconosciuto: $t" ;;
    esac
done
