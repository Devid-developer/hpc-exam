# Ottimizzazione dello stencil 2D seriale e OpenMP

## 1. Scopo e contesto

Il programma simula la diffusione di una quantità di energia su una griglia bidimensionale mediante uno stencil a cinque punti. A ogni iterazione il valore interno della cella `(i,j)` viene calcolato usando il valore centrale e i quattro vicini nelle direzioni nord, sud, est e ovest.

Questa versione nasce dal template seriale fornito per l'esame e costituisce una base comune per due modalità di esecuzione:

- esecuzione single-core, impostando un solo thread OpenMP;
- esecuzione multicore, compilando con OpenMP e scegliendo il numero di thread.

I benchmark finali saranno eseguiti sulla partizione Booster di Leonardo CINECA. Ogni nodo Booster ha una CPU Intel Xeon Platinum 8358 Ice Lake single-socket con 32 core fisici, 512 GiB di memoria DDR4 3200 MHz e quattro GPU NVIDIA A100, che non vengono usate da questa implementazione CPU. La campagna di benchmark deve quindi limitare OpenMP a 32 thread fisici per nodo e registrare sempre nodo, compilatore, opzioni di compilazione e binding. La descrizione aggiornata dell'hardware è disponibile nella [documentazione CINECA di Leonardo](https://docs.hpc.cineca.it/hpc/leonardo.html).

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

È riportata anche una stima di bandwidth con un modello da 24 byte per aggiornamento. Questo valore non rappresenta la somma ingenua di cinque load e uno store, ma assume riuso dei dati attraverso la gerarchia di cache e uno specifico modello di traffico verso memoria. Deve quindi essere presentato come modello, non come misura hardware. La futura analisi dovrà confrontarlo con contatori prestazionali e con la bandwidth sostenibile del nodo.

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

Per Leonardo le opzioni specifiche dell'architettura e del compilatore saranno definite durante la fase di benchmark. Un flag come `-march=native` può produrre codice adatto al nodo su cui avviene la compilazione, ma va usato sui compute node appropriati e non indiscriminatamente sui login node o tra partizioni con microarchitetture diverse.

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

Le ottimizzazioni principali seguono quattro idee fondamentali del corso: ridurre il lavoro nel loop caldo, esporre al compilatore accessi regolari e assenza di alias, rispettare la gerarchia di memoria e distribuire soltanto iterazioni realmente indipendenti. La misura quantitativa del beneficio di ciascun intervento verrà aggiunta nella successiva fase di benchmark su Leonardo.
