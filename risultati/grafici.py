# grafici.py
#
# Leggo i file CSV salvati da esegui_test.sh nella cartella dati/ e faccio
# i grafici del capitolo dei risultati, che salvo nella cartella grafici/.
# Ogni prova e' ripetuta piu' volte: nei grafici uso la media delle ripetizioni.
# Stampo anche la tabella dei tempi medi a numero di iterazioni fisso.
#
# Uso:  python3 grafici.py      (serve matplotlib)

import csv
import os

import matplotlib.pyplot as plt

os.makedirs("grafici", exist_ok=True)


def leggi(nome):
    """Legge dati/<nome>.csv e restituisce una lista di righe (dizionari)."""
    with open(f"dati/{nome}.csv") as f:
        return list(csv.DictReader(f))


def media(righe, colonne, campo):
    """Raggruppa le righe che hanno gli stessi valori nelle colonne indicate
    (cioe' le ripetizioni della stessa prova) e fa la media del campo.
    Restituisce un dizionario: (valori delle colonne) -> media."""
    valori = {}
    for r in righe:
        chiave = tuple(r[c] if c == "metodo" else int(r[c]) for c in colonne)
        valori.setdefault(chiave, []).append(float(r[campo]))
    return {k: sum(v) / len(v) for k, v in valori.items()}


# ---------------------------------------------------------------------------
# Grafico 1: iterazioni al variare di N
# ---------------------------------------------------------------------------
it_jacobi = media(leggi("jacobi_N"), ["N"], "iterazioni")
N = sorted(k[0] for k in it_jacobi)

plt.figure(figsize=(6, 4))
plt.loglog(N, [it_jacobi[(n,)] for n in N], "o-", label="Jacobi")

it_schwarz = media(leggi("schwarz_N_p"), ["N", "p"], "iterazioni")
for p in [2, 4, 8]:
    N_p = sorted(k[0] for k in it_schwarz if k[1] == p)
    plt.loglog(N_p, [it_schwarz[(n, p)] for n in N_p], "s-", label=f"Schwarz, p = {p}")

plt.xlabel("N (nodi interni)")
plt.ylabel("iterazioni (tol = $10^{-6}$)")
plt.grid(True, which="major", alpha=0.3)
plt.legend()
plt.savefig("grafici/iterazioni_N.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------------------
# Grafico 2: Schwarz, iterazioni al variare della sovrapposizione
# ---------------------------------------------------------------------------
it_sovr = media(leggi("sovrapposizione"), ["p", "delta"], "iterazioni")

plt.figure(figsize=(6, 4))
for p in [2, 4]:
    delta = sorted(k[1] for k in it_sovr if k[0] == p)
    plt.semilogy(delta, [it_sovr[(p, d)] for d in delta], "o-", label=f"p = {p}")

plt.xlabel("$\\ell$ (strati di sovrapposizione per lato)")
plt.ylabel("iterazioni (tol = $10^{-6}$)")
plt.title("Schwarz, N = 2048")
plt.grid(True, which="major", alpha=0.3)
plt.legend()
plt.savefig("grafici/overlap.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------------------
# Grafico 3: tempo medio al variare di p, con un numero di iterazioni fisso
# ---------------------------------------------------------------------------
tempo = media(leggi("tempi_processi"), ["metodo", "N", "p"], "tempo")
valori_N = sorted({k[1] for k in tempo})

fig, assi = plt.subplots(1, 2, figsize=(9, 3.8))
print("tempo medio (s) a numero di iterazioni fisso")
print("metodo   N         p=1       p=2       p=4")
for asse, metodo in zip(assi, ["jacobi", "schwarz"]):
    for n in valori_N:
        tempi = [tempo[(metodo, n, p)] for p in [1, 2, 4]]
        asse.plot([1, 2, 4], tempi, "o-", label=f"N = {n}")
        print(f"{metodo:8} {n:<9} {tempi[0]:<9.3f} {tempi[1]:<9.3f} {tempi[2]:.3f}")
    asse.set_xticks([1, 2, 4])
    asse.set_xlabel("p (processi)")
    asse.set_title("Jacobi" if metodo == "jacobi" else "Schwarz ($\\ell = 0$)")
    asse.set_ylim(bottom=0)
    asse.set_ylabel("tempo medio (s)")
    asse.grid(True, alpha=0.3)
assi[1].legend(fontsize=8)
plt.savefig("grafici/tempi_processi.png", dpi=200, bbox_inches="tight")
