# Piano operativo per benchmark, report e slide

## 1. Obiettivo degli esperimenti

Il report deve rispondere separatamente a tre domande:

1. Le ottimizzazioni single-core rendono più veloce il kernel a parità di algoritmo e compilatore?
2. La versione OpenMP scala all'aumentare dei core all'interno di un nodo?
3. La versione MPI scala all'aumentare dei processi e dei nodi, e quanto costa la comunicazione degli halo?

Mescolare queste tre domande in un unico eseguibile renderebbe difficile attribuire un miglioramento o un rallentamento alla causa corretta.

## 2. Quale eseguibile usare

### 2.1 Esperimenti seriali

Per confrontare il codice iniziale e le ottimizzazioni single-core bisogna usare il codice seriale, senza runtime MPI.

Le versioni da confrontare sono:

- `serial_baseline`: codice iniziale, corretto ma non ottimizzato;
- `serial_optimized`: codice seriale finale con kernel semplificato, `restrict`, puntatori di riga, allocazione allineata e altre ottimizzazioni;
- facoltativamente `openmp_optimized` con `OMP_NUM_THREADS=1`, per misurare l'overhead introdotto dal runtime OpenMP.

Non conviene usare il programma MPI con un solo task come baseline seriale. Anche con un task, la versione MPI crea il comunicatore, effettua lo scambio degli halo e usa una diversa struttura del dominio. Si misurerebbe quindi `seriale + overhead MPI`, non il vero codice seriale.

Il confronto corretto è:

```text
baseline seriale              vs seriale ottimizzato
seriale ottimizzato           vs OpenMP con 1 thread
OpenMP con 1 thread           vs OpenMP con P thread
MPI con 1 rank e 1 thread     vs MPI con R rank e 1 thread
```

Per calcolare lo speedup OpenMP si deve usare come riferimento la versione OpenMP con un thread. La differenza tra seriale ottimizzato e OpenMP a un thread va mostrata separatamente come overhead OpenMP.

Per calcolare lo speedup MPI si deve usare come riferimento la versione MPI con un rank. La versione seriale resta comunque utile per mostrare l'overhead assoluto di MPI a un rank.

### 2.2 Esperimenti OpenMP

Usare un solo processo Slurm e variare `--cpus-per-task` e `OMP_NUM_THREADS`:

```text
--nodes=1
--ntasks=1
--cpus-per-task=P
OMP_NUM_THREADS=P
```

Su un nodo Booster di Leonardo è presente un singolo socket Intel Xeon Platinum 8358 Ice Lake con 32 core fisici e 512 GiB di memoria DDR4. Il nodo misurato espone però due domini NUMA interni al socket, rispettivamente sui core `0-15` e `16-31`, effetto del Sub-NUMA Clustering. Il massimo OpenMP naturale è 32 thread, non 112, ma binding e first touch devono comunque considerare entrambi i domini NUMA.

Una sequenza consigliata è:

```text
1, 2, 4, 8, 16, 24, 32 thread
```

- `16` misura metà dei core del socket;
- `32` misura il socket e il nodo completi;
- `24` aggiunge un punto nella regione in cui la bandwidth di memoria potrebbe essere già prossima alla saturazione.

Non eseguire OpenMP a 64 o 112 thread sul Booster: supererebbe i 32 core fisici e misurerebbe oversubscription o hardware threads, non uno scaling corretto sui core.

Mantenere esplicita l'affinità, per esempio:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun --cpu-bind=cores ./stencil_serial_final_omp_O3 ...
```

`close` è un buon punto di partenza per sfruttare la località. Un confronto `close` contro `spread` può essere interessante, ma va considerato un esperimento secondario e non mescolato con quello sullo scheduler.

Sul Booster il confronto è particolarmente rilevante. Con `close` e non più di 16 thread l'allocazione può usare principalmente un solo dominio NUMA; con `spread`, se la CPU mask assegnata al processo comprende tutti i 32 core, i thread possono essere distribuiti sui due domini e sfruttare più bandwidth. Per un confronto controllato conviene riservare al processo OpenMP l'intera CPU mask da 32 core, variare soltanto `OMP_NUM_THREADS` e confrontare `close`/`spread` mantenendo identico il first touch.

### 2.3 Esperimenti MPI e ibridi

Prima misurare MPI puro:

```text
OMP_NUM_THREADS=1
--cpus-per-task=1
rank MPI = 1, 2, 4, 8, 16, 32, ...
```

Questo isola la scalabilità distribuita e il costo dello scambio degli halo.

Sul Booster al massimo 32 rank single-thread possono occupare i 32 core fisici di un nodo. I punti fino a 32 rank misurano quindi MPI intra-node; 64 rank richiedono normalmente due nodi e introducono la rete InfiniBand. Per dimostrare davvero la componente distributed-memory è opportuno includere almeno `64` rank, ed eventualmente `128`, oppure limitare esplicitamente i rank per nodo in modo da attraversare prima il confine di rete. Nei grafici va indicato chiaramente dove cambia il numero di nodi, perché è normale osservare una discontinuità.

Solo successivamente studiare la versione ibrida MPI+OpenMP. Non è corretto prendere automaticamente il numero di thread migliore di OpenMP e usarlo per ogni rank. Per esempio, due rank da 32 thread non possono condividere un nodo Booster da 32 core.

Per confrontare diverse decomposizioni ibride a parità di risorse su un nodo si può mantenere:

```text
rank_per_node * thread_per_rank = 32
```

Esempi validi:

```text
1 x 32
2 x 16
4 x 8
8 x 4
16 x 2
32 x 1
```

Non è necessario provarli tutti. Una selezione ragionevole è `1x32`, `2x16`, `4x8`, `8x4` e `32x1`. Dopo aver scelto una configurazione ibrida, la si mantiene fissa per nodo e si varia il numero di nodi.

Se l'esame basic non richiede esplicitamente una versione ibrida, è sufficiente presentare bene OpenMP puro e MPI puro; l'ibrido può essere una breve estensione finale.

## 3. Versioni del codice da conservare

Non è necessario produrre decine di versioni. È più efficace conservare pochi checkpoint interpretabili:

### 3.1 Single-core

1. `S0`: template iniziale corretto, senza ottimizzazioni manuali.
2. `S1`: espressione stencil semplificata e invarianti spostate fuori dal loop.
3. `S2`: versione finale con accessi per riga, `restrict`, `size_t` e memoria allineata.

Se la differenza tra `S1` e `S2` è minima, nelle slide si possono unire e mostrare soltanto `S0` contro `S2`. Nel report è comunque utile elencare le trasformazioni.

Tutte le versioni confrontate devono essere compilate con lo stesso compilatore e gli stessi flag. Altrimenti non si può distinguere il beneficio del codice da quello del compilatore.

Separatamente si può mostrare un piccolo studio dei flag:

```text
-O0, -O2, -O3, eventualmente -O3 -march=native
```

Questo studio non va confuso con il confronto tra versioni algoritmiche.

### 3.2 OpenMP

Conservare:

1. versione con `schedule(static)`;
2. versione con `schedule(dynamic)`;
3. facoltativamente `schedule(guided)`.

Lo stencil esegue quasi lo stesso lavoro per ogni riga, quindi ci si aspetta che `static` sia migliore: assegna il lavoro una volta sola, ha overhead ridotto e mantiene una distribuzione stabile delle righe. `dynamic` è utile quando le iterazioni hanno costi irregolari, condizione che qui non si verifica.

Non serve confrontare gli scheduler per ogni numero di thread. È sufficiente usare alcuni casi rappresentativi, ad esempio:

```text
1, 8, 16 e 32 thread
```

con la stessa griglia e lo stesso numero di iterazioni. Una volta verificato che `static` è la scelta migliore, tutte le curve di scaling devono usare esclusivamente quella versione.

### 3.3 MPI

Per un corso basic è consigliabile mantenere la progressione didattica:

1. scambio bloccante;
2. scambio non bloccante con `MPI_Isend`/`MPI_Irecv` e `MPI_Waitall`;
3. overlap tra comunicazione e calcolo soltanto se è stato trattato nel corso.

Un confronto onesto deve distinguere:

```text
blocking
nonblocking + wait immediato
nonblocking + vero overlap
```

Usare chiamate non bloccanti e fare immediatamente `MPI_Waitall` evita alcuni problemi di deadlock, ma non sovrappone comunicazione e calcolo. Non bisogna dichiarare di aver ottenuto overlap se il calcolo inizia solo dopo il completamento delle comunicazioni.

Con `MPI_Send` e `MPI_Recv` bisogna evitare il deadlock. Le alternative basic sono:

- ordinare send e receive diversamente per rank pari e dispari;
- usare `MPI_Sendrecv`, se ammesso;
- effettuare gli scambi una direzione alla volta con un protocollo sicuro.

Il file parallelo attualmente creato usa topologia cartesiana, datatype derivato per le colonne e comunicazioni non bloccanti. Prima della consegna verrà semplificato in base agli strumenti effettivamente ammessi dal corso, per esempio usando vicini calcolati esplicitamente e packing manuale delle colonne.

Il confronto blocking/nonblocking può essere limitato a pochi punti, ad esempio `2, 8, 16, 32` rank. Scelta la variante migliore e didatticamente accettabile, solo quella viene usata per strong e weak scaling.

## 4. Strong scaling

Nel strong scaling la dimensione globale del problema e il numero di iterazioni restano fissi mentre aumentano le risorse.

### 4.1 OpenMP strong scaling

```text
stessa griglia globale
stesso numero di iterazioni
stesse sorgenti, seed e condizioni al contorno
thread = 1 ... 32
un solo nodo
```

Formule:

```text
Speedup(P)    = T(1) / T(P)
Efficiency(P) = Speedup(P) / P
```

Usare `T(1)` dell'eseguibile OpenMP, non quello del seriale puro. Il seriale puro viene mostrato come confronto separato.

### 4.2 MPI strong scaling

```text
stessa griglia globale
stesso numero di iterazioni
OMP_NUM_THREADS=1 per MPI puro
rank = 1, 2, 4, 8, 16, 32, ...
```

Formule:

```text
Speedup(R)    = T_MPI(1) / T_MPI(R)
Efficiency(R) = Speedup(R) / R
```

Per la versione ibrida, il numero totale di core è:

```text
C = rank MPI * thread OpenMP per rank
```

e l'efficienza va calcolata rispetto a `C`, specificando chiaramente la configurazione di riferimento.

Lo strong scaling smetterà inevitabilmente di migliorare quando ogni rank/thread riceve poco lavoro e il costo di sincronizzazione, parallel region e comunicazione diventa dominante. Questo non è un fallimento: è uno dei risultati principali da spiegare.

## 5. Weak scaling

Nel weak scaling il lavoro per unità di elaborazione deve restare circa costante.

### 5.1 OpenMP weak scaling

Si mantiene approssimativamente costante:

```text
numero di celle globale / numero di thread
```

Per conservare anche la forma quadrata della griglia, i punti più puliti sono numeri quadrati di thread:

```text
1, 4 e 16
```

Se il dominio base è `S0 x S0`, si può usare:

```text
S(P) = S0 * sqrt(P)
```

nelle due direzioni. Per numeri non quadrati si scelgono dimensioni rettangolari il cui prodotto sia circa `P*S0*S0`, documentando la scelta.

### 5.2 MPI weak scaling

Ogni rank deve mantenere una patch locale approssimativamente costante. Se la griglia dei processi è `Px x Py` e la patch locale è `Lx x Ly`, la griglia globale diventa:

```text
global_x = Px * Lx
global_y = Py * Ly
```

La metrica principale è:

```text
Weak efficiency(R) = T(1) / T(R)
```

Un valore vicino a uno è ideale. È utile mostrare anche il rapporto comunicazione/calcolo, perché nel weak scaling la quantità di dati per rank resta costante ma aumenta il numero complessivo di comunicazioni e possono emergere effetti di rete.

## 6. Dimensione del problema e durata delle run

La griglia deve essere abbastanza grande da non misurare principalmente overhead di timer, startup OpenMP o startup MPI. Inoltre deve superare chiaramente le cache quando si vuole studiare il comportamento memory-bound.

Procedura consigliata:

1. scegliere una griglia iniziale, per esempio `10000 x 10000`;
2. usare `-o 0`, perché stampa e I/O falsano i tempi;
3. aumentare il numero di iterazioni finché anche la configurazione più veloce dura almeno alcuni secondi;
4. mantenere poi esattamente gli stessi parametri per tutti i punti della stessa curva.

Non usare una run molto breve da pochi millisecondi. Il rumore del sistema e gli overhead fissi potrebbero essere della stessa grandezza del tempo misurato.

Per tutte le prove mantenere fissi:

```text
numero e posizione delle sorgenti
energia per sorgente
seed o modalità -F
condizioni periodiche
frequenza di iniezione
numero di iterazioni
```

salvo quando la dimensione globale deve cambiare per il weak scaling.

## 7. Ripetizioni e statistica

Ogni configurazione dovrebbe essere ripetuta almeno cinque volte; sette o dieci ripetizioni sono preferibili se il costo lo consente.

Per ogni punto salvare tutte le ripetizioni, non soltanto il valore migliore. Nel report usare:

- mediana come valore centrale robusto;
- minimo come indicazione della prestazione ottenibile con poco rumore;
- deviazione standard o intervallo min-max come variabilità.

Non scegliere manualmente una singola run favorevole. Se la prima esecuzione è sistematicamente diversa, eseguire una run di warm-up esplicitamente esclusa dalla statistica e documentarlo.

## 8. Cosa salvare per ogni run

### 8.1 Metadati indispensabili

Ogni riga del dataset deve contenere:

```text
timestamp
Slurm job ID
hostname o lista dei nodi
partizione e QoS
numero di nodi
numero totale di task MPI
task MPI per nodo
thread OpenMP per task
CPU assegnate per task
OMP_PLACES e OMP_PROC_BIND
versione del codice
commit Git o hash del sorgente
compilatore e versione
moduli caricati
flag di compilazione
dimensione x e y
numero di iterazioni
numero di sorgenti
seed/modalità sorgenti
periodicità
tipo di scheduler OpenMP
tipo di comunicazione MPI
indice della ripetizione
```

Senza questi dati una misura potrebbe non essere riproducibile.

### 8.2 Tempi da salvare

Salvare almeno:

1. `t_wall`: tempo end-to-end del ciclo computazionale;
2. `t_update`: tempo del kernel stencil;
3. `t_inject`: tempo di iniezione delle sorgenti;
4. `t_energy`: tempo della riduzione energetica locale/globale;
5. `t_comm`: per MPI, tempo di scambio degli halo;
6. `t_setup`: facoltativo, inizializzazione MPI, decomposizione e allocazione.

La metrica principale per lo scaling deve essere `t_wall`, perché rappresenta il tempo realmente percepito per completare il calcolo. `t_update` serve a capire se il kernel scala; `t_comm` spiega perché MPI smette di scalare; `t_inject` e `t_energy` mostrano se funzioni secondarie diventano colli di bottiglia.

Per MPI non usare il tempo del solo rank 0. La durata parallela è determinata dal rank più lento. Per ogni regione temporizzata è utile ridurre tra i rank:

```text
minimo, media e massimo
```

Il massimo è quello da usare per prestazioni e scaling. La differenza tra massimo e media misura lo sbilanciamento.

### 8.3 Metriche derivate

Dal tempo salvato si calcolano dopo le run:

```text
GLUP/s
GFLOP/s
bandwidth modellata
speedup
parallel efficiency
communication fraction = t_comm / t_wall
update fraction        = t_update / t_wall
imbalance              = (t_max - t_avg) / t_avg
```

È meglio salvare i tempi grezzi e calcolare queste quantità in fase di analisi. In questo modo, se cambia una formula, non occorre ripetere le simulazioni.

### 8.4 Correttezza

Salvare anche:

```text
energia totale iniettata
energia finale del sistema
```

Con bordi periodici i due valori devono essere compatibili entro la tolleranza floating-point. Con bordi non periodici è normale che l'energia del sistema sia minore, perché il bordo agisce come pozzo.

L'energia è una metrica di correttezza, non di prestazione. Serve a dimostrare che una versione più veloce non ha cambiato il risultato fisico.

### 8.5 File grezzi da conservare

Per ogni job conservare:

- script `sbatch` usato;
- stdout e stderr originali;
- CSV prodotto dal programma;
- output di `scontrol show job $SLURM_JOB_ID`;
- dati di accounting ottenuti con `sacct`;
- file con `module list`, compilatore e flag;
- commit Git o copia esatta dei sorgenti compilati.

I job di produzione devono essere eseguiti sui compute node tramite Slurm, non sui login node. La [documentazione ufficiale CINECA sullo scheduler](https://docs.hpc.cineca.it/hpc/hpc_scheduler.html) raccomanda `sbatch` per le run di produzione e `srun` all'interno dell'allocazione.

## 9. Organizzazione dei risultati

Una struttura semplice è:

```text
results/
  environment.txt
  serial/
    baseline/
    optimized/
    omp_one_thread/
  openmp/
    scheduler/
    strong/
    weak/
  mpi/
    communication/
    strong/
    weak/
  hybrid/
```

Un nome file informativo può essere:

```text
omp_strong_t032_rep03_job123456.csv
mpi_blocking_r016_t001_rep02_job123789.csv
hybrid_n002_r004_t008_rep05_job124000.csv
```

Non sovrascrivere mai i risultati di una ripetizione precedente.

## 10. Grafici consigliati

### Slide single-core

- barre del tempo `S0`, `S1`, `S2`;
- speedup rispetto a `S0`;
- breve elenco delle trasformazioni;
- eventualmente report di vettorizzazione del compilatore.

### Slide OpenMP

- tempo contro numero di thread;
- speedup reale e linea ideale;
- parallel efficiency;
- confronto static/dynamic in un grafico piccolo;
- evidenziare il nodo completo a 32 core e l'eventuale saturazione della bandwidth prima di tale punto.

### Slide MPI

- tempo e speedup contro numero di rank;
- efficienza;
- frazione `t_comm/t_wall`;
- confronto blocking/nonblocking;
- indicazione del numero di nodi e della griglia MPI.

### Slide weak scaling

- tempo contro risorse, con linea ideale orizzontale;
- weak efficiency;
- dimensione globale associata a ogni punto.

Per speedup e scaling usare l'asse x logaritmico in base due quando i punti sono potenze di due. Non inserire troppe curve nello stesso grafico: ogni figura deve rispondere a una domanda precisa.

## 11. Sequenza pratica consigliata

1. Congelare `serial_baseline` e `serial_optimized`.
2. Calibrare dimensione e iterazioni per ottenere run di alcuni secondi.
3. Eseguire il confronto single-core con almeno cinque ripetizioni.
4. Misurare l'overhead OpenMP confrontando seriale ottimizzato e OpenMP a un thread.
5. Confrontare `static` e `dynamic` soltanto su pochi thread rappresentativi.
6. Eseguire strong scaling OpenMP con `static`.
7. Eseguire weak scaling OpenMP.
8. Semplificare il codice MPI agli strumenti consentiti dal corso.
9. Verificare MPI a un rank contro il risultato seriale.
10. Confrontare blocking e nonblocking su pochi numeri di rank.
11. Eseguire strong scaling MPI con la variante scelta.
12. Eseguire weak scaling MPI.
13. Valutare l'ibrido soltanto dopo aver compreso separatamente OpenMP e MPI.
14. Generare grafici dai CSV grezzi e costruire le slide attorno alle conclusioni osservate.

Questa sequenza evita di spendere budget su una matrice completa prima di avere scelto dimensione del problema, scheduler e variante di comunicazione corretti.
