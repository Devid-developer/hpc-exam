# Handoff per nuova chat Codex

## Contesto

Repository:

```text
/home/dev_developer/Universita/Magistrale/hpc-exam
```

GitHub:

```text
https://github.com/Devid-developer/hpc-exam.git
branch: main
ultimo commit pubblicato: d8ed600 Reorganize serial and OpenMP benchmarks
```

Obiettivo: progetto d'esame Basic HPC magistrale. Ottimizzazione di uno stencil
2D a 5 punti, con benchmark sul nodo Booster di Leonardo CINECA.

La parte seriale e OpenMP è sostanzialmente conclusa. Il prossimo lavoro è la
versione MPI.

## Richieste iniziali

- Non ottimizzare realmente `stencil_template_serial.c/.h`: è la baseline e
  basta che compili.
- Le versioni `stencil_serial_final.c/.h` erano considerate definitive.
- Riordinare gli script Slurm.
- Usare:

```make
CFLAGS = -O3 -Wall -Wextra -march=native -fopenmp -Iinclude -g
ARGS = -x 25000 -y 25000 -n 200 -o 0
```

- Usare sorgenti casuali, senza `-F`.
- Salvare per ogni run in un unico CSV per script:

```text
run_name
t_wall
t_get_total_energy
t_update_plane
t_inject_energy
glups
```

- Tutti i job Slurm devono avere limite di 30 minuti.

## Modifiche effettuate

### Makefile

Il `Makefile` produce:

```text
build/stencil_template_serial_O1
build/stencil_template_serial_O3
build/stencil_serial_final_O1
build/stencil_serial_final_O3
build/stencil_serial_final_omp_O3
```

Target principali:

```bash
make all
make template-serial
make final-serial
make omp-serial
```

Decisione metodologica importante:

- le build seriali rimuovono `-fopenmp`, per essere realmente seriali;
- il binario OpenMP è separato;
- `CFLAGS` mantiene comunque esattamente le flag richieste.

Questo permette di distinguere seriale puro, OpenMP con un thread e OpenMP con
più thread.

### Script Slurm

Sono presenti:

```text
go_template_serial.sh
go_final_serial.sh
go_omp_serial.sh
```

Tutti hanno:

```bash
#SBATCH --time=00:30:00
#SBATCH --account=IscrB_SPIESMD
#SBATCH --partition=boost_usr_prod
```

`go_template_serial.sh` esegue template O1 e template O3.

`go_final_serial.sh` esegue il finale seriale puro O1 e O3.

`go_omp_serial.sh` esegue:

```text
strong scaling: 1 2 4 8 16 24 32 thread
binding: close e spread
weak scaling: 1 2 4 8 16 32 thread
BASE_SIDE predefinito: 5000
```

“sparse” nella richiesta originale è stato interpretato come `spread`, perché
`sparse` non è un valore standard di `OMP_PROC_BIND`.

OpenMP usa:

```bash
OMP_DYNAMIC=false
OMP_PLACES=cores
OMP_PROC_BIND=close/spread
```

Ogni job compila in una directory privata:

```text
results/<job-id>_*/build/
```

Questo evita conflitti se vengono sottomessi più job contemporaneamente.

Ogni script produce un solo CSV:

```text
run_name,t_wall,t_get_total_energy,t_update_plane,t_inject_energy,glups
```

Sono conservati anche raw log, `job.out`, `job.err`, `build.log`,
`environment.txt`, `lscpu.txt`, `slurm_job.txt`, `sacct.txt` e
`source_changes.patch`.

`REPEATS` è configurabile e vale 1 per default. Per misure definitive è
consigliato:

```bash
sbatch --export=ALL,REPEATS=5 go_omp_serial.sh
```

### Baseline template

In `src/stencil_template_serial.c` sono stati aggiunti soltanto:

- timer monotono;
- accumulo dei quattro tempi richiesti;
- calcolo GLUP/s;
- output leggibile e riga CSV.

L'algoritmo baseline non è stato ripulito o ottimizzato.

In `include/stencil_template_serial.h` è stato aggiunto l'include guard. La
baseline continua a generare diversi warning con `-Wall -Wextra`; sono
intenzionali e accettati perché il requisito era soltanto che compilasse.

### Pulizia repository

Eliminati i vecchi script:

```text
run_booster_benchmarks.sh
run_booster_complete_no_mpi.sh
run_booster_no_mpi.sh
run_booster_weak_openmp.sh
```

Aggiornati:

```text
.gitignore
documentazione_opt.md
piano_benchmark_report.md
```

`results/`, `build/` e gli output Slurm sono ignorati.

La rimozione di `src/stencil_parallel_final.c` era già presente prima del
riordino ed è stata inclusa nel commit su richiesta di pubblicare tutto.

## Verifiche già effettuate

Build completa:

```bash
make clean all
```

Risultato: successo. Rimangono solo i warning attesi della baseline.

Sono state fatte run locali con griglia `100x100`, 5 iterazioni e sorgenti
casuali nelle seguenti configurazioni:

```text
template O1
template O3
finale seriale O1
finale seriale O3
OpenMP O3 con 1 thread
OpenMP O3 con 2 thread
OpenMP O3 con 4 thread
```

Tutte hanno restituito `exit code 0` e prodotto timer e CSV validi.

È stato eseguito anche un controllo con bordi periodici (`-p 1`). Template,
finale seriale e OpenMP hanno restituito:

```text
injected energy = 5
system energy   = 5
```

Quindi l'energia viene conservata correttamente con bordi periodici.

I tre script sono stati anche eseguiti end-to-end con un ambiente Slurm
simulato e dimensioni ridotte: parsing e generazione CSV funzionano.

La working tree era pulita e sincronizzata con GitHub dopo il push. La creazione
di questo file `HANDOFF.md` è una modifica successiva e potrebbe non essere
ancora committata.

## Nota sulle sorgenti casuali

Gli script non passano `-F`.

Template e finale non generano necessariamente le stesse posizioni casuali:

- il template usa lo stato predefinito di `lrand48`;
- il finale esegue `srand48(seed)`, con seed predefinito 0.

Perciò con bordi non periodici l'energia finale può differire tra template e
finale senza indicare necessariamente un errore, perché le sorgenti possono
trovarsi a distanze diverse dal bordo.

Per una futura verifica MPI cella per cella bisogna garantire le stesse sorgenti.
Soluzione suggerita:

1. rank 0 genera tutte le sorgenti casuali con seed noto;
2. le coordinate vengono distribuite agli altri rank;
3. ogni rank conserva quelle appartenenti alla propria patch.

## Prossimo obiettivo: MPI

Conviene iniziare ora la versione parallela partendo da:

```text
src/stencil_template_parallel.c
include/stencil_template_parallel.h
```

Approccio consigliato per Basic HPC:

1. MPI puro, con `OMP_NUM_THREADS=1`.
2. Prima correttezza, poi prestazioni.
3. Inizialmente usare `MPI_Sendrecv`, se ammesso dal corso.
4. Evitare inizialmente ibrido MPI+OpenMP, datatype complessi, nonblocking e
   overlap.
5. Implementare decomposizione del dominio, patch locali con halo, gestione dei
   resti, vicini periodici/non periodici, scambio degli halo, distribuzione
   delle sorgenti ed energia globale con `MPI_Reduce`.
6. Testare con 1, 2 e 4 rank, griglia `100x100`, griglia `101x97` e bordi sia
   periodici sia non periodici.
7. Aggiungere i timer MPI:

```text
t_wall
t_update_plane
t_inject_energy
t_get_total_energy
t_exchange_halos
GLUP/s
```

Per le prestazioni parallele bisogna usare il massimo tempo tra i rank,
ottenuto con:

```c
MPI_Reduce(..., MPI_MAX, ...)
```

## Quando sarà disponibile Leonardo

Sottomettere prima smoke test con `REPEATS=1`:

```bash
sbatch go_template_serial.sh
sbatch go_final_serial.sh
sbatch go_omp_serial.sh
```

Controllare compilazione e CSV. Solo successivamente passare a `REPEATS=5`.

La priorità della prossima chat dovrebbe essere creare una versione MPI
minimale, corretta e facilmente spiegabile, senza modificare ulteriormente le
versioni seriale/OpenMP salvo bug reali.

## Prompt suggerito per la nuova chat

```text
Leggi HANDOFF.md e usalo come contesto operativo. Controlla lo stato attuale
della repository e inizia a implementare la versione MPI minimale partendo dai
file stencil_template_parallel, senza modificare la parte seriale/OpenMP.
Prima proponimi brevemente la decomposizione del dominio e il protocollo di
scambio degli halo.
```
