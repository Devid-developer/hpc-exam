# Ottimizzazione dello stencil 2D seriale, OpenMP, MPI e ibrido

## 1. Scopo e contesto

Il programma simula la diffusione di una quantità di energia su una griglia bidimensionale mediante uno stencil a cinque punti. A ogni iterazione il valore interno della cella `(i,j)` viene calcolato usando il valore centrale e i quattro vicini nelle direzioni nord, sud, est e ovest.

Questa versione nasce dai template seriale e parallelo forniti per l'esame e
costituisce una base comune per quattro modalità di esecuzione:

- esecuzione seriale senza runtime OpenMP;
- esecuzione OpenMP su memoria condivisa;
- esecuzione MPI a memoria distribuita;
- esecuzione ibrida MPI+OpenMP.

I benchmark finali sono stati eseguiti sulla partizione Booster di Leonardo
CINECA. Ogni nodo Booster ha una CPU Intel Xeon Platinum 8358 Ice Lake
single-socket con 32 core fisici, 512 GiB di memoria DDR4 3200 MHz e quattro
GPU NVIDIA A100, che non vengono usate da questa implementazione CPU. La
campagna ha quindi limitato OpenMP a 32 thread fisici per nodo e ha registrato
nodo, compilatore, opzioni di compilazione e binding. La descrizione aggiornata
dell'hardware è disponibile nella [documentazione CINECA di Leonardo](https://docs.hpc.cineca.it/hpc/leonardo.html).

I coefficienti scelti sono parte della specifica del progetto:

```c
ALPHA    = 0.6
C_CENTER = 0.6
C_NEIGH  = 0.25 * (1.0 - ALPHA) = 0.1
```

Pertanto il kernel applica:

```text
new(i,j) = 0.6 old(i,j)
         + 0.1 [old(i-1,j) + old(i+1,j) + old(i,j-1) + old(i,j+1)]
```

La somma dei pesi è `0.6 + 4*0.1 = 1`. In assenza di sorgenti e con condizioni periodiche, l'operatore non crea né distrugge energia; con bordi non periodici gli halo restano a zero e rappresentano un pozzo di calore esterno.

## 2. Struttura dei dati e double buffering

Il dominio fisico ha dimensione `xsize * ysize`. Ogni piano viene allocato con un halo di una cella su ogni lato, quindi le dimensioni effettive sono:

```text
(xsize + 2) * (ysize + 2)
```

L'indice lineare è:

```c
IDX(i,j) = (size_t)j * (xsize + 2) + i;
```

Sono mantenuti due piani, `OLD` e `NEW`. Durante un'iterazione si legge esclusivamente da `OLD` e si scrive in `NEW`; alla fine vengono scambiati gli indici logici. Il double buffering evita copie complete della griglia e, soprattutto, elimina dipendenze read-after-write all'interno dello stesso aggiornamento. Ogni cella della nuova griglia dipende solo dallo stato dell'iterazione precedente, rendendo indipendenti gli aggiornamenti e quindi adatti sia alla vettorizzazione SIMD sia alla parallelizzazione OpenMP.

## 3. Confronto con il template iniziale

### 3.1 Semplificazione algebrica del kernel

Il template calcolava separatamente `result`, `sum_i` e `sum_j` e ripeteva divisioni e operazioni su `1-alpha` dentro il doppio ciclo. La nuova versione calcola una volta sola i coefficienti e usa una singola espressione:

```c
result[i] = cc * center[i]
          + cn * (north[i] + south[i] + center[i-1] + center[i+1]);
```

La trasformazione è algebricamente equivalente con `alpha=0.6`, ma riduce il numero di temporanei e rende esplicite al compilatore le operazioni invarianti rispetto al ciclo. Il principio teorico è la *loop-invariant code motion*: ciò che non cambia con `i` o `j` va calcolato fuori dal loop caldo. Un compilatore ottimizzante può eseguire autonomamente parte di questa trasformazione, ma una formulazione semplice rende più facile l'analisi, la vettorizzazione e il mantenimento del codice.

Il modello adottato conta sei operazioni floating-point per cella: quattro addizioni per combinare i termini e due moltiplicazioni. Eventuali istruzioni FMA generate dal compilatore possono ridurre il numero di istruzioni hardware, ma non cambiano il conteggio algoritmico convenzionale dei FLOP.

### 3.2 Accesso per righe e riduzione dell'aritmetica degli indici

Nel template ogni accesso espandeva la macro `IDX`, ripetendo la moltiplicazione `j*fxsize`. La versione ottimizzata costruisce all'inizio di ogni riga quattro puntatori:

```c
center = old + j*fxsize;
north  = old + (j-1)*fxsize;
south  = old + (j+1)*fxsize;
result = new + j*fxsize;
```

Il ciclo interno usa quindi offset semplici. Poiché `i` è l'indice più interno, gli accessi sono contigui in memoria. Questo ordine è coerente con il layout row-major del C e favorisce:

- località spaziale;
- prefetch hardware;
- uso efficiente delle cache line;
- generazione di load/store SIMD contigui.

### 3.3 Puntatori `restrict`

I puntatori ai piani e alle righe sono qualificati con `restrict`. Il programmatore garantisce così che, durante il kernel, gli oggetti letti tramite `old` non siano modificati tramite `new`. Senza questa informazione il compilatore deve considerare un possibile alias tra input e output e può essere costretto a produrre versioni multiple del loop o a rinunciare ad alcune trasformazioni.

La validità di `restrict` deriva direttamente dal double buffering: i due piani sono allocazioni distinte e non si sovrappongono.

### 3.4 Eliminazione di `register`

Il template usava ripetutamente il qualificatore storico `register`. Nei compilatori moderni l'allocazione dei registri è decisa dall'ottimizzatore sulla base della rappresentazione intermedia e della pressione sui registri. Il qualificatore non costituisce una strategia di ottimizzazione utile ed è stato rimosso per rendere il codice più chiaro.

### 3.5 Tipi degli indici

Le espressioni che producono offset lineari usano `size_t`. Questo è il tipo naturale per dimensioni e offset di oggetti in memoria e impedisce che il prodotto tra riga e stride venga eseguito nel dominio più piccolo di `int` prima della conversione.

## 4. Allocazione e inizializzazione della memoria

### 4.1 Due allocazioni distinte

Il template eseguiva una sola `malloc` grande il doppio di un piano e faceva puntare `NEW` nella seconda metà. La funzione di rilascio liberava soltanto il puntatore iniziale. La nuova versione alloca esplicitamente due piani distinti e li libera separatamente.

Questa scelta rende evidente la proprietà di non aliasing richiesta da `restrict`, semplifica la gestione degli errori parziali e permette di trattare i due buffer come oggetti indipendenti.

### 4.2 Allineamento a 64 byte

`posix_memalign` sostituisce `malloc` e allinea l'inizio di ciascun piano a 64 byte:

```c
posix_memalign(&plane, 64, frame * sizeof(double));
```

Sessantaquattro byte corrispondono alla dimensione tipica di una cache line sulle CPU Intel considerate. L'allineamento non garantisce che ogni riga sia allineata, perché lo stride include i due elementi di halo, ma garantisce una base nota e rimuove possibili penalità all'inizio dell'allocazione. L'effetto reale deve essere confermato con i benchmark e con il report di vettorizzazione del compilatore.

### 4.3 Inizializzazione e first touch

Entrambi i piani vengono inizializzati a zero. Quando OpenMP è attivo, anche l'inizializzazione è suddivisa staticamente tra i thread. Sui sistemi NUMA la politica *first touch* assegna normalmente una pagina fisica al dominio NUMA del core che la tocca per primo. Parallelizzare l'inizializzazione con una distribuzione coerente con quella usata nel kernel può quindi migliorare l'affinità della memoria nelle esecuzioni multicore.

Per l'esecuzione a un thread il medesimo codice resta seriale. Sul Booster è comunque importante impostare correttamente il pinning. Le misure sul nodo `lrdn1549` mostrano un solo socket fisico suddiviso mediante Sub-NUMA Clustering in due domini NUMA: core `0-15` e core `16-31`. Non esiste un confine tra socket, ma esiste quindi un confine NUMA interno al socket. First touch e affinità devono distribuire pagine e thread coerentemente sui due domini. Con `OMP_PROC_BIND=close`, fino a 16 thread il calcolo può restare concentrato su un dominio e sfruttare soltanto parte della bandwidth; usando più di 16 core entra in gioco anche il secondo dominio.

## 5. Parallelizzazione OpenMP

Il ciclo esterno sulle righe è parallelizzato con:

```c
#pragma omp parallel for schedule(static)
```

Ogni iterazione del ciclo scrive una riga distinta di `NEW` e legge solamente `OLD`; non esistono quindi race condition tra righe. `schedule(static)` assegna blocchi deterministici di iterazioni ai thread con overhead ridotto. Il carico per riga è uniforme, quindi uno scheduling dinamico non offrirebbe un vantaggio di bilanciamento e introdurrebbe costo di runtime aggiuntivo.

Le direttive sono racchiuse in `#ifdef _OPENMP`. La stessa base sorgente può essere compilata senza OpenMP, senza warning per pragma sconosciuti, oppure con `-fopenmp`. I test single-core della build OpenMP vengono eseguiti impostando un thread, mentre i test multicore variano `OMP_NUM_THREADS`.

Il calcolo dell'energia totale usa una riduzione:

```c
#pragma omp parallel for schedule(static) reduction(+:tot)
```

Ogni thread accumula un totale privato; il runtime combina i parziali alla fine. Questo evita una race sulla variabile condivisa e non richiede una sezione critica per ogni elemento. L'ordine delle somme floating-point può cambiare al variare del numero di thread, quindi sono possibili differenze negli ultimi bit pur in presenza di un risultato numericamente corretto.

## 6. Condizioni al contorno e sorgenti

Gli halo sono necessari affinché il kernel interno non contenga branch per distinguere celle interne e celle di bordo. Con bordi non periodici gli halo rimangono zero; con bordi periodici, dopo ogni aggiornamento:

- l'ultima riga fisica viene copiata nell'halo superiore;
- la prima riga fisica viene copiata nell'halo inferiore;
- l'ultima colonna fisica viene copiata nell'halo sinistro;
- la prima colonna fisica viene copiata nell'halo destro.

Spostare la gestione dei bordi fuori dal loop caldo è un esempio di *loop unswitching* manuale a livello algoritmico: il test `periodic` non viene eseguito per ogni cella.

L'iniezione aggiorna anche gli halo corrispondenti quando una sorgente si trova su un bordo periodico. In questo modo l'energia appena iniettata è visibile attraverso il bordo già nell'aggiornamento successivo.

La versione iniziale effettuava inoltre un'iniezione prima del loop quando la frequenza era maggiore di uno, ma non aggiungeva tale quantità a `injected_heat`. Questa iniezione non contabilizzata è stata rimossa: ora ogni iniezione avviene nel loop ed è registrata nello stesso punto.

Sono state aggiunte due modalità utili alla riproducibilità:

- sorgenti pseudocasuali con seed selezionabile tramite `-s`;
- sorgenti deterministiche nelle posizioni a un quarto e tre quarti del dominio tramite `-F`.

Gli script di benchmark usano la modalità pseudocasuale predefinita: non passano
`-F`. La modalità deterministica resta disponibile soltanto per i test di
correttezza.

## 7. Parsing e validazione della riga di comando

Per mantenere il parsing semplice e vicino al template iniziale, la conversione degli argomenti usa `atoi`, `atol` e `atof`. Dopo la conversione vengono comunque controllati dominio positivo, numero positivo di iterazioni, numero non negativo di sorgenti ed energia non negativa.

Come nel template, una frequenza maggiore di uno viene limitata a uno. Il valore `-f 0` significa iniezione a ogni iterazione. Questa soluzione privilegia la semplicità, ma non distingue una stringa non numerica da un vero valore zero e non diagnostica esplicitamente overflow o caratteri finali: tali limiti sono accettati per questa versione dell'esercizio.

## 8. Pulizia dell'header

L'header iniziale conteneva numerosi include non necessari, costanti relative a MPI non usate dal seriale e prototipi pubblici di `initialize` e `memory_release` che non corrispondevano più alle funzioni private implementate nel sorgente finale.

Sono stati introdotti:

- include guard `STENCIL_SERIAL_FINAL_H`;
- il solo include richiesto direttamente, `<stddef.h>` per `size_t`;
- linkage `static inline` per le funzioni definite nell'header;
- rimozione delle dichiarazioni obsolete e delle costanti inutilizzate;
- protezione condizionale delle direttive OpenMP.

`static inline` evita i problemi del modello di linkage delle funzioni `inline` del C quando lo stesso header è incluso da unità di traduzione diverse. Ogni unità riceve una definizione interna che il compilatore può integrare; l'inlining rimane comunque una decisione dell'ottimizzatore, non un obbligo imposto dalla keyword.

## 9. Correzione del dump

Il template attraversava il buffer come se non esistessero halo: usava `size[0]` come stride e partiva dalla prima cella dell'allocazione. Il risultato non corrispondeva quindi alla regione fisica della griglia.

La nuova implementazione:

- usa lo stride reale `xsize+2`;
- parte da `(1,1)` e scrive soltanto le celle fisiche;
- converte ogni riga da `double` a `float` in un buffer temporaneo;
- apre il file con modalità binaria `wb`;
- gestisce gli errori principali di apertura e allocazione.

Il formato resta una sequenza row-major di `xsize*ysize` valori `float`, senza halo.

## 10. Timing e metriche prestazionali

È stato aggiunto un timer monotono basato su `clock_gettime(CLOCK_MONOTONIC)`. Vengono misurati separatamente:

- tempo del kernel `update_plane`;
- tempo di iniezione;
- tempo della riduzione energetica;
- wall time complessivo.

Separare il tempo del kernel dal resto evita che I/O e diagnostica alterino la misura centrale. Dal numero di aggiornamenti

```text
updates = xsize * ysize * niterations
```

si calcolano:

```text
GLUP/s  = updates / t_update / 10^9
GFLOP/s = 6 * updates / t_update / 10^9
```

È riportata anche una stima di bandwidth con un modello da 24 byte per
aggiornamento. Questo valore non rappresenta la somma ingenua di cinque load e
uno store, ma assume riuso dei dati attraverso la gerarchia di cache e uno
specifico modello di traffico verso memoria. Viene quindi presentato come
modello analitico, non come misura ottenuta da contatori hardware.

Una riga CSV facilita la raccolta automatica dei risultati senza dover estrarre dati dall'output descrittivo.

## 11. Makefile e modalità di compilazione

Il Makefile definisce target distinti per template e versione finale, a entrambi
i livelli di ottimizzazione richiesti:

```text
make all               -> tutte le build seriali e OpenMP
make template-serial   -> template -O1 e -O3
make final-serial      -> finale -O1 e -O3
make omp-serial        -> binario finale OpenMP -O3 dedicato
make run-template-o1  -> esecuzione del target template -O1
make run-final-o3     -> esecuzione del target finale -O3
make run-omp          -> esempio OpenMP
```

La variabile `CFLAGS` contiene `-O3 -Wall -Wextra -march=native -fopenmp
-Iinclude -g`. I target seriali rimuovono `-fopenmp` e selezionano `-O1` oppure
`-O3`; il target OpenMP mantiene invece `-fopenmp`. In questo modo il confronto
seriale non include il runtime OpenMP, mentre `go_omp_serial.sh` misura anche la
build OpenMP con un thread. `-O3` abilita trasformazioni aggressive sui loop e
la vettorizzazione automatica.

Su Leonardo la compilazione dei benchmark avviene nel job sul compute node. Il
flag `-march=native` produce quindi codice per la microarchitettura del Booster;
lo stesso binario non va riutilizzato indiscriminatamente su partizioni con
microarchitetture differenti.

## 12. Sintesi degli improvement

Rispetto ai file iniziali, la versione finale introduce:

1. kernel algebricamente compatto con coefficienti invarianti estratti dal loop;
2. accesso row-major tramite puntatori di riga;
3. qualificatori `restrict` per facilitare la vettorizzazione;
4. offset `size_t`;
5. allocazioni separate e allineate a 64 byte;
6. inizializzazione compatibile con il first touch OpenMP;
7. parallelizzazione statica del kernel;
8. riduzione OpenMP dell'energia;
9. gestione degli halo periodici fuori dal loop caldo;
10. rimozione dell'iniezione iniziale non contabilizzata;
11. sorgenti riproducibili tramite seed o posizioni fisse;
12. parsing semplice, coerente con il template, e validazione dei parametri principali;
13. header autocontenuto e privo di API obsolete;
14. dump corretto della sola regione fisica;
15. timing separati, GLUP/s, GFLOP/s, stima della bandwidth e output CSV;
16. Makefile con build seriale e OpenMP riproducibili.

Le ottimizzazioni principali seguono quattro idee fondamentali del corso:
ridurre il lavoro nel loop caldo, esporre al compilatore accessi regolari e
assenza di alias, rispettare la gerarchia di memoria e distribuire soltanto
iterazioni realmente indipendenti. Le sezioni successive estendono queste idee
alla memoria distribuita e riportano le misure raccolte su Leonardo.

## 13. Estensione MPI e decomposizione del dominio

La versione MPI è implementata in `src/stencil_parallel_final.c` e riusa lo
stesso kernel locale ottimizzato della versione OpenMP. La differenza
fondamentale è che ogni rank conserva in memoria soltanto un blocco rettangolare
del dominio globale, completo dei propri halo. Non viene quindi replicata
l'intera griglia su ogni processo.

### 13.1 Griglia bidimensionale dei processi

I rank sono organizzati logicamente in una griglia `Px * Py`. Poiché il pool di
primitive ammesso per il progetto è limitato, non vengono usati
`MPI_Cart_create` o `MPI_Cart_shift`. La fattorizzazione viene scelta
manualmente provando tutti i divisori del numero di processi e minimizzando il
rapporto tra i lati del blocco locale:

```text
local_x = global_x / Px
local_y = global_y / Py
score   = max(local_x/local_y, local_y/local_x)
```

Una forma locale vicina al quadrato riduce il rapporto tra perimetro, che
determina il volume di comunicazione, e area, che determina il lavoro utile.
Il mapping tra rank lineare e coordinate logiche è row-major:

```text
coord_x = rank % Px
coord_y = rank / Px
```

Per esempio, 32 rank su una griglia quadrata producono una decomposizione
`4 * 8`, mentre 64 rank producono `8 * 8`.

### 13.2 Domini non divisibili

La decomposizione non richiede che le dimensioni globali siano divisibili per
la griglia dei processi. Indicando con `N` una dimensione globale, con `P` il
numero di blocchi e con `c` la coordinata del rank, la dimensione e l'offset
locali sono:

```text
base      = N / P
remainder = N % P
size(c)   = base + (c < remainder)
offset(c) = c*base + min(c, remainder)
```

I primi `remainder` blocchi ricevono quindi una cella aggiuntiva. Questa scelta
mantiene il massimo sbilanciamento pari a una sola riga o colonna ed è stata
verificata anche sul dominio non divisibile `101 * 97`.

### 13.3 Individuazione manuale dei vicini

Ogni rank calcola direttamente i vicini nord, sud, est e ovest. Con condizioni
non periodiche un bordo esterno è rappresentato da `MPI_PROC_NULL`; le relative
operazioni di comunicazione vengono omesse e l'halo rimane a zero. Con
condizioni periodiche il vicino oltre il bordo è il rank sul lato opposto.

Quando una direzione della griglia dei processi ha dimensione uno, non viene
eseguita una comunicazione MPI con se stessi. Gli halo periodici di quella
direzione vengono aggiornati localmente dal kernel. Gli angoli non devono
essere comunicati perché lo stencil a cinque punti usa soltanto i quattro
vicini ortogonali.

## 14. Scambio degli halo

All'inizio di ogni iterazione il piano corrente contiene lo stato prodotto
dall'iterazione precedente. Prima del calcolo vengono aggiornati i quattro halo
con una comunicazione nearest-neighbour.

Le righe nord e sud sono già contigue nel layout row-major e possono essere
inviate direttamente dal piano. Le colonne est e ovest non sono invece
contigue; per evitare `MPI_Type_vector`, non disponibile nel pool scelto, sono
copiate esplicitamente in buffer contigui di send e receive. Packing e unpacking
delle colonne sono parallelizzati con OpenMP quando sono presenti più thread.

La sequenza dello scambio è:

1. packing delle colonne fisiche est e ovest;
2. pubblicazione delle ricezioni con `MPI_Irecv`;
3. pubblicazione degli invii con `MPI_Isend`;
4. completamento con un unico `MPI_Waitall`;
5. unpacking delle colonne ricevute negli halo.

Tag distinti identificano la direzione del messaggio e impediscono
accoppiamenti ambigui. Pubblicare prima le ricezioni e usare primitive
non-bloccanti evita dipendenze dall'ordine di esecuzione dei rank e possibili
deadlock.

Nell'implementazione corrente non c'è sovrapposizione tra comunicazione e
calcolo: `MPI_Waitall` termina prima di `update_plane`. Le primitive
non-bloccanti sono quindi usate per rendere sicuro e simmetrico lo scambio, non
per nascondere la latenza. Una possibile evoluzione sarebbe calcolare prima la
regione interna mentre i messaggi sono in volo e completare in seguito le
quattro fasce di bordo; tale trasformazione renderebbe però il kernel più
complesso e non è necessaria per gli obiettivi dell'esame.

## 15. Sorgenti, riduzioni e correttezza globale

Le sorgenti casuali vengono generate soltanto dal rank zero usando il seed
richiesto. L'elenco delle coordinate globali viene distribuito con
`MPI_Bcast`; ogni rank seleziona poi le sorgenti comprese nel proprio intervallo
e le converte in coordinate locali, includendo l'offset dell'halo. Non sono
necessari `Scatterv` o strutture dati differenti tra i rank.

L'energia locale è calcolata con la riduzione OpenMP descritta nella sezione 5.
I valori locali vengono poi sommati sul rank zero con:

```c
MPI_Reduce(..., MPI_SUM, 0, MPI_COMM_WORLD);
```

Anche gli errori di allocazione devono essere trattati collettivamente: se un
solo rank fallisse e uscisse mentre gli altri proseguono, il programma potrebbe
bloccarsi alla comunicazione successiva. La funzione `global_max_int` realizza
un OR logico mediante `MPI_Reduce` con `MPI_MAX`, seguito da `MPI_Bcast` per
rendere noto l'esito a tutti. Questo sostituisce intenzionalmente
`MPI_Allreduce`.

Il test smoke ha eseguito domini periodici e non periodici `101 * 97` con 1, 2,
4 e 8 rank. Dopo 80 iniezioni:

- nel caso periodico l'energia globale è risultata esattamente 80 con tutte le
  decomposizioni;
- nel caso non periodico il valore è risultato 79.9051 con tutte le
  decomposizioni; la quantità inferiore a 80 è attesa perché gli halo esterni a
  zero rappresentano un pozzo di calore.

L'indipendenza del risultato dal numero di rank verifica congiuntamente
ownership delle sorgenti, offset dei blocchi, scambio degli halo e riduzione
globale. I tempi del test smoke sono troppo piccoli per avere valore
prestazionale e vengono usati soltanto per la correttezza.

## 16. Modello ibrido MPI+OpenMP

Il programma viene inizializzato con:

```c
MPI_Init_thread(..., MPI_THREAD_FUNNELED, &level_obtained);
```

Il livello `MPI_THREAD_FUNNELED` è sufficiente perché tutte le chiamate MPI
avvengono al di fuori delle regioni parallele e sono effettuate dal thread
principale. OpenMP parallelizza il calcolo locale, l'inizializzazione, il
packing e l'unpacking delle colonne e la somma dell'energia. Non è quindi
necessario richiedere il più costoso `MPI_THREAD_MULTIPLE`.

Sul Booster ogni configurazione ibrida mantiene costante il numero complessivo
di core e varia il rapporto tra rank MPI e thread OpenMP. Su un nodo sono state
provate:

```text
1*32, 2*16, 4*8, 8*4, 16*2, 32*1
```

Su due nodi i rank totali raddoppiano, mantenendo lo stesso layout per nodo:

```text
2*32, 4*16, 8*8, 16*4, 32*2, 64*1
```

La prima cifra indica i rank totali e la seconda i thread per rank. Gli script
impostano `OMP_PLACES=cores`, `OMP_PROC_BIND=spread` e `OMP_DYNAMIC=false`; Slurm
assegna un blocco di core a ciascun rank. In questo modo il runtime non cambia
dinamicamente il numero di thread e ogni configurazione usa rispettivamente 32
o 64 core fisici.

## 17. Primitive MPI utilizzate

La versione finale usa un insieme deliberatamente ristretto di primitive:

- `MPI_Init_thread`, `MPI_Comm_rank`, `MPI_Comm_size` e `MPI_Finalize` per il
  ciclo di vita;
- `MPI_Barrier` e `MPI_Wtime` per sincronizzazione iniziale e timing;
- `MPI_Bcast` per sorgenti e propagazione degli errori;
- `MPI_Reduce` con `MPI_SUM` o `MPI_MAX` per energia, errori e tempi;
- `MPI_Irecv`, `MPI_Isend` e `MPI_Waitall` per gli halo.

Non vengono usate topologie cartesiane, comunicazioni collettive avanzate,
datatype derivati o comunicazioni one-sided. La semplicità del pool rende
espliciti decomposizione e movimento dei dati, che sono i concetti principali
richiesti dall'esercizio.

## 18. Timing distribuito e metodologia dei benchmark

Prima della regione temporizzata viene eseguita una `MPI_Barrier`, quindi ogni
rank usa `MPI_Wtime`. Alla fine i cinque tempi locali vengono ridotti con
`MPI_MAX`:

- `t_update`: solo kernel locale;
- `t_comm`: packing, comunicazione e unpacking degli halo;
- `t_inject`: iniezione locale;
- `t_energy`: riduzione dell'energia;
- `t_wall`: intera simulazione temporizzata.

Il massimo è la metrica corretta perché l'esecuzione globale termina alla
velocità del rank più lento. I massimi delle diverse metriche possono provenire
da rank differenti e i timer di comunicazione e aggiornamento sono riportati
separatamente; non devono quindi essere sommati a posteriori come se fossero
partizioni perfette dello stesso tempo massimo.

I GLUP/s MPI sono calcolati dagli aggiornamenti globali divisi per il massimo
di `t_update`. Il wall time resta la metrica usata per speedup ed efficienza,
perché include anche la comunicazione:

```text
strong speedup:     S(p) = T(1) / T(p)
strong efficiency:  E(p) = S(p) / p
weak efficiency:    E(p) = T(1) / T(p)
```

Tutti i job usano nodi esclusivi, core fisici, compilazione `-O3
-march=native -fopenmp`, limite di 30 minuti e sorgenti pseudocasuali. Ogni
directory dei risultati conserva CSV, output raw, informazioni Slurm, `lscpu`,
ambiente software, commit Git e patch locale.

La campagna comprende:

- OpenMP strong scaling a griglia fissa `25000 * 25000`, con binding `close` e
  `spread`;
- OpenMP weak scaling con 25 milioni di celle per thread;
- MPI strong scaling sulla griglia fissa `25000 * 25000`, da 1 a 64 rank;
- MPI weak scaling con 25 milioni di celle per rank;
- sweep ibrido a problema fisso su uno e due nodi;
- weak scaling ibrido da `25000 * 25000` su un nodo a `50000 * 25000` su due
  nodi, mantenendo 19,531,250 celle per core.

## 19. Risultati OpenMP

### 19.1 Strong scaling e binding

Con un thread entrambe le politiche partono da circa 152.16 s e 0.826 GLUP/s.
`spread` è chiaramente preferibile fino a 16 thread:

| Thread | close [s] | spread [s] | spread [GLUP/s] |
|------:|----------:|-----------:|-----------------:|
| 1  | 152.144 | 152.164 | 0.826 |
| 2  | 84.466  | 77.634  | 1.619 |
| 4  | 54.171  | 43.310  | 2.900 |
| 8  | 40.068  | 27.057  | 4.638 |
| 16 | 32.882  | 19.198  | 6.530 |
| 24 | 21.597  | 21.787  | 5.749 |
| 32 | 16.067  | 16.095  | 7.782 |

Il vantaggio di `spread` fino a 16 thread è coerente con la suddivisione SNC
del processore: distribuire i thread permette di usare prima entrambi i domini
NUMA e una frazione maggiore della bandwidth. Il punto a 24 thread è non
monotono e ricorda che binding, first touch e contesa della memoria possono
contare più del solo numero di core. Poiché ogni punto principale contiene una
sola ripetizione, le differenze piccole non devono essere sovrainterpretate.

### 19.2 Weak scaling

Il dominio cresce da `5000 * 5000` con un thread a `40000 * 20000` con 32
thread, conservando 25 milioni di celle per thread. Il tempo passa da 5.753 s a
20.908 s e il throughput totale da 0.874 a 7.668 GLUP/s. L'efficienza weak
globale è quindi circa il 27.5%.

Il risultato non indica uno sbilanciamento del loop, che assegna lo stesso
numero di celle a ogni thread. Mostra invece che i thread di uno stesso nodo
condividono la bandwidth DRAM: aumentando le risorse di calcolo non cresce in
proporzione la banda di memoria disponibile. Lo stencil raggiunge quindi il
regime memory-bound.

## 20. Risultati MPI

### 20.1 Strong scaling

| Rank | Nodi | Wall time [s] | t_comm [s] | GLUP/s |
|-----:|-----:|--------------:|-----------:|-------:|
| 1  | 1 | 152.269 | 0.000 | 0.825 |
| 2  | 1 | 86.640  | 0.575 | 1.450 |
| 4  | 1 | 53.375  | 0.452 | 2.359 |
| 8  | 1 | 39.277  | 0.535 | 3.201 |
| 16 | 1 | 32.583  | 0.577 | 3.867 |
| 32 | 1 | 16.168  | 0.466 | 7.812 |
| 64 | 2 | 8.056   | 0.680 | 15.799 |

Da 1 a 64 rank lo speedup complessivo sul wall time è circa `18.9x`. La
scalabilità rispetto al singolo core non è ideale perché il kernel è limitato
dalla memoria e tutti i rank di un nodo condividono lo stesso sottosistema
DRAM. Il confronto più significativo per l'estensione multinodo è però 32
contro 64 rank: raddoppiando nodi, core e rank, il tempo passa da 16.168 s a
8.056 s, corrispondente a uno speedup di `2.007x`. La comunicazione inter-node
rimane abbastanza piccola da non impedire il dimezzamento del tempo.

Le prestazioni MPI e OpenMP su 32 core sono quasi identiche: 7.812 GLUP/s per
MPI e circa 7.78 GLUP/s per OpenMP. Ciò indica che packing e halo exchange
intra-node non dominano il kernel per questa dimensione globale.

### 20.2 Weak scaling

Ogni rank conserva 25 milioni di celle. Il wall time cresce da 5.710 s con un
rank a circa 21 s con 32 rank sullo stesso nodo, ancora a causa della bandwidth
condivisa. Il confronto tra nodi a pieno carico è invece quasi ideale:

```text
32 rank, 1 nodo: 20.969 s,  7.712 GLUP/s
64 rank, 2 nodi: 21.007 s, 15.412 GLUP/s
```

L'efficienza weak da uno a due nodi pieni è `20.969 / 21.007 = 99.8%`, mentre
il throughput raddoppia quasi esattamente. Ogni nodo aggiuntivo porta con sé un
nuovo sottosistema di memoria; questo spiega perché la scalabilità fra nodi è
molto migliore della scalabilità aumentando i soli core all'interno del nodo.

## 21. Risultati ibridi MPI+OpenMP

### 21.1 Sweep a problema fisso

Sulla griglia `25000 * 25000` tutte le configurazioni a 32 core impiegano circa
15.9 s, mentre tutte quelle a 64 core impiegano circa 8.0 s:

| Rank per nodo * thread | 1 nodo [s] | 2 nodi [s] |
|-----------------------:|-------------:|-------------:|
| 1 * 32 | 15.957 | 8.035 |
| 2 * 16 | 15.965 | 8.006 |
| 4 * 8  | 15.901 | 8.010 |
| 8 * 4  | 15.891 | 8.019 |
| 16 * 2 | 15.898 | 7.969 |
| 32 * 1 | 15.921 | 7.963 |

La variazione tra layout è inferiore all'uno per cento. A questa scala il
vantaggio di ridurre il numero di messaggi usando più thread è compensato dal
costo del runtime OpenMP e dal packing di colonne più lunghe. Il caso puro MPI
è marginalmente il più veloce, ma la differenza è troppo piccola e basata su
una sola ripetizione per affermare una superiorità generale. Il risultato
principale è la robustezza: tutti i rapporti MPI/OpenMP sfruttano quasi allo
stesso modo i core disponibili.

### 21.2 Weak scaling ibrido

Nel weak scaling ibrido il dominio raddoppia insieme al numero di nodi e di
core. Ogni layout per nodo viene quindi confrontato con il proprio equivalente
su due nodi. I wall time rimangono sempre compresi tra 15.83 e 15.98 s.

| Rank per nodo * thread | Efficienza 1 -> 2 nodi |
|-----------------------:|-------------------------:|
| 1 * 32 | 99.62% |
| 2 * 16 | 99.65% |
| 4 * 8  | 99.88% |
| 8 * 4  | 99.40% |
| 16 * 2 | 99.77% |
| 32 * 1 | 99.68% |

Tutte le configurazioni ottengono quindi efficienza prossima al 100%. Il
layout `4 * 8` presenta il valore migliore nella misura corrente, ma la
differenza rispetto agli altri casi è nell'ordine di pochi centesimi di
secondo. È più corretto concludere che, per questo problema, la scalabilità
multinodo è sostanzialmente indipendente dal mix MPI/OpenMP scelto.

## 22. Makefile, job Slurm e riproducibilità MPI

Il target:

```text
make mpi
```

compila `src/stencil_parallel_final.c` mediante `mpicc`, usando gli stessi flag
`-O3 -Wall -Wextra -march=native -fopenmp -Iinclude -g` della versione OpenMP.
Gli script dedicati sono:

- `go_mpi_smoke.sh`: correttezza con più rank, bordi periodici e dominio non
  divisibile;
- `go_mpi_strong.sh`: strong scaling MPI da 1 a 64 rank;
- `go_mpi_weak.sh`: weak scaling MPI a 25 milioni di celle per rank;
- `go_mpi_hybrid.sh`: sweep del rapporto rank/thread a problema fisso;
- `go_mpi_hybrid_weak.sh`: weak scaling ibrido accoppiato fra uno e due nodi.

Ogni script produce un unico CSV con nome della run, configurazione di risorse,
tempi massimi, GLUP/s ed energie iniettata e misurata. Gli output raw vengono
mantenuti per poter verificare il parsing e la conservazione dell'energia. Il
caricamento dell'ambiente usa `module purge` prima di OpenMPI, evitando il
conflitto osservato tra moduli GCC differenti durante il primo smoke test.

## 23. Conclusioni della parte parallela

L'implementazione distribuita mantiene il kernel locale semplice e sposta la
complessità nei confini tra sottodomini. Le scelte principali sono:

1. decomposizione 2D bilanciata, anche per dimensioni non divisibili;
2. vicini e periodicità calcolati senza topologie MPI avanzate;
3. righe comunicate direttamente e colonne impacchettate senza datatype
   derivati;
4. riduzioni globali limitate a energia, errori e timing;
5. modello `MPI_THREAD_FUNNELED`, coerente con l'uso ibrido;
6. confronto sistematico tra OpenMP, MPI puro e configurazioni ibride.

I risultati mostrano due regimi distinti. All'interno di un nodo la scalabilità
è limitata soprattutto dalla bandwidth di memoria, comune a thread e rank. Tra
nodi pieni, invece, ogni nodo aggiunge memoria e bandwidth: strong e weak
scaling risultano quasi ideali da uno a due nodi. Per il dominio studiato il
rapporto tra MPI e OpenMP ha un effetto secondario rispetto al numero totale di
core e nodi, e tutte le configurazioni ibride raggiungono efficienza weak
prossima al 100%.
