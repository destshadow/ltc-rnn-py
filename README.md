# LTC RNN · Python

Progetto sperimentale in Python e PyTorch per costruire una rete ricorrente con dinamica a capacità e conduttanze sinaptiche, ispirata alle Liquid Time-Constant networks.

Il progetto comprende una cella LTC, un solver semi-implicito con validazione degli ingressi, l'elaborazione di sequenze e un classificatore con uscita lineare. Sono disponibili dati sintetici sull'ordine temporale, script di controllo e visualizzazioni delle sinapsi e dello stato. Il ciclo di addestramento e la valutazione su dati separati non sono ancora implementati.

## Dinamica e forme dei tensori

Ogni neurone ha capacità, conduttanza di perdita e potenziale di riposo. Ogni collegamento ha intensità, pendenza, soglia e potenziale di inversione. Capacità, conduttanza di perdita, intensità e pendenza sono ottenute tramite `softplus` con un piccolo termine positivo.

La conduttanza attiva di un collegamento è:

```text
g = strength * sigmoid(slope * (source - threshold))
```

Il solver aggiorna lo stato con:

```text
v_next = ((C / h) * v + g_leak * E_rest + sum(g * E_rev))
         / (C / h + g_leak + sum(g))
```

Qui `h = dt / substeps`; le somme comprendono sinapsi sensoriali e ricorrenti. Durante un intervallo `dt`, gli ingressi restano costanti e le conduttanze ricorrenti vengono ricalcolate a ogni sottopasso.

| Tensore | Forma |
| --- | --- |
| Ingressi | `[batch, input_size]` |
| Stato | `[batch, hidden_size]` |
| Sequenze | `[batch, istanti, input_size]` |
| Storia degli stati | `[batch, istanti, hidden_size]` |
| Logits del classificatore | `[batch, num_classes]` |
| Parametri sinaptici | `[source_size, target_size]` |
| Conduttanze attive | `[batch, source_size, target_size]` |
| Drive e conduttanza totale | `[batch, target_size]` |

Il modello differisce dal [progetto C++](https://github.com/destshadow/ltc-rnn-cpp), che usa un obiettivo `tanh`, una costante di tempo appresa ed Eulero esplicito. Non è una traduzione numericamente equivalente.

## Ambiente

La configurazione locale usa un ambiente virtuale `.venv` e le seguenti librerie, installate dall'autore:

| Libreria | Configurazione |
| --- | --- |
| PyTorch | `2.13.0`, indice delle wheel CUDA `cu130` |
| NumPy | Versione non fissata |
| Matplotlib | Versione non fissata |

La GPU rilevata nell'ambiente locale tramite `nvidia-smi` è una **NVIDIA GeForce RTX 3060 con 12 GB di VRAM**. L'accesso alla GPU da PyTorch non è ancora stato verificato.

## Preparazione dell'ambiente

I comandi seguenti sono per Bash su Linux o WSL e riportano la configurazione usata dall'autore. Richiedono Git, Python 3 con supporto a `venv` e, per l'uso della GPU, un ambiente NVIDIA configurato.

### 1. Clonare il repository

```bash
git clone https://github.com/destshadow/ltc-rnn-py.git
cd ltc-rnn-py
```

### 2. Creare e attivare l'ambiente virtuale

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installare le librerie

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy matplotlib
```

Questi sono i comandi di installazione comunicati dall'autore; l'installazione non è stata ripetuta per verificarne la riproducibilità. Non è ancora presente un file che fissi tutte le versioni delle dipendenze.

### 4. Controllare interprete e GPU

```bash
python -c "import sys; print(sys.executable)"
nvidia-smi
```

Il percorso dell'interprete deve terminare con `.venv/bin/python`. `nvidia-smi` verifica che la GPU sia visibile al sistema; il suo esito non conferma da solo l'utilizzo di CUDA da parte di PyTorch.

Per uscire dall'ambiente virtuale:

```bash
deactivate
```

## Verifica di PyTorch e CUDA

Con l'ambiente virtuale attivo, lo script disponibile si esegue con:

```bash
python check_environment.py
```

Lo script stampa le versioni di PyTorch e CUDA, verifica la disponibilità della GPU ed esegue una moltiplicazione su un tensore CUDA. Se PyTorch non riesce ad accedere alla GPU, solleva un errore. Il risultato atteso della moltiplicazione è `[2.0, 4.0, 6.0]`; l'esecuzione non è ancora stata verificata dall'assistente.

## Controlli disponibili

Eseguire dalla radice del repository con l'ambiente virtuale attivo:

```bash
python check_config.py
python check_environment.py
python check_state.py
python check_neuron_parameters.py
python check_synapse_parameters.py
python check_synapses.py
python check_solver.py
python check_solver_validation.py
python check_solver_dynamics.py
python check_cell.py
python check_temporal_order.py
python check_classifier.py
```

`check_config.py`, `check_solver_validation.py`, `check_solver_dynamics.py` e `check_temporal_order.py` funzionano su CPU. `check_cell.py` aggiunge un confronto CPU/GPU quando CUDA è disponibile; `check_classifier.py` sceglie CUDA se disponibile, altrimenti CPU. Gli altri script elencati richiedono CUDA nella loro versione attuale. Questa lista documenta i controlli disponibili, non attesta che siano stati tutti eseguiti e superati.

| Script | Verifica |
| --- | --- |
| `check_environment.py` | Versioni, accesso alla GPU e operazione su tensore CUDA |
| `check_config.py` | Creazione della configurazione e rifiuto di una dimensione non valida |
| `check_state.py` | Forma, azzeramento e indipendenza degli stati |
| `check_neuron_parameters.py` | Numero dei parametri, positività e gradienti |
| `check_synapse_parameters.py` | Forme, positività e gradienti dei parametri sinaptici |
| `check_synapses.py` | Apertura al 50% sulla soglia, indipendenza dei batch e gradienti verso le sorgenti |
| `check_solver.py` | Decadimento senza sinapsi e confronto con la soluzione analitica |
| `check_solver_validation.py` | Rifiuto di batch, dimensioni, tipi numerici e durata non validi nei casi coperti |
| `check_solver_dynamics.py` | Riferimenti scalari, equivalenza dei sottopassi, stato immutato e gradienti temporali |
| `check_cell.py` | Composizione della cella, stato iniziale e confronto CPU/GPU opzionale |
| `check_temporal_order.py` | Struttura delle coppie, etichette e riproducibilità dei dati |
| `check_classifier.py` | Logits, perdita e gradienti nella cella e nello strato di uscita |

Il test del solver verifica che cento sottopassi siano più accurati di un singolo passo nel caso di decadimento. Non costituisce una verifica completa della dinamica ricorrente o dell'addestramento.

## Sequenze e classificazione

`LTCCell` raggruppa i parametri dei neuroni e le sinapsi sensoriali e ricorrenti. Lo stato iniziale segue dispositivo e tipo numerico della cella. `run_sequence` elabora ingressi `[batch, istanti, ingressi]` e restituisce storia e stato finale, mantenendo il percorso dei gradienti. Accetta anche uno stato iniziale esplicito.

`SequenceClassifier` applica `nn.Linear` allo stato finale e restituisce logits utilizzabili con `CrossEntropyLoss`. `check_classifier.py` calcola una perdita e chiama `backward()`, ma non aggiorna i pesi: le previsioni stampate sono quelle del modello non addestrato.

`make_temporal_order(pairs, seed=...)` genera `2 * pairs` sequenze di 30 passi, ciascuna con due impulsi opposti. La classe 0 presenta prima l'impulso positivo, la classe 1 quello negativo. Ogni coppia condivide posizioni e ampiezza; gli ultimi campioni sono sempre zero. Il seme rende riproducibile la generazione.

Il compito richiede conservare informazione dagli impulsi precedenti, ma può essere risolto ricordando il segno dell'ultimo impulso non nullo: non misura da solo capacità temporali generali.

## Visualizzazioni

```bash
python -m visualization.synapse_viewer
```

Il grafico mostra la conduttanza di una sinapsi al variare della sorgente. Tre slider modificano soglia, pendenza e intensità. La visualizzazione usa la CPU e richiede un ambiente grafico con un backend interattivo di Matplotlib; va avviata come modulo dalla radice del repository.

Per osservare la risposta dello stato a un impulso:

```bash
python -m visualization.state_viewer
```

Il secondo viewer usa una cella non addestrata su CPU e mostra ingresso, stati dei neuroni e distanza tra la traiettoria con impulso e quella con ingresso nullo. Richiede anch'esso un backend grafico interattivo.

## Struttura

| File | Responsabilità |
| --- | --- |
| `ltc/config.py` | Configurazione immutabile e validazione delle dimensioni |
| `ltc/state.py` | Creazione dello stato iniziale |
| `ltc/neuron_parameters.py` | Parametri apprendibili dei neuroni |
| `ltc/synapse_parameters.py` | Parametri apprendibili dei collegamenti |
| `ltc/synapses.py` | Conduttanze, drive e somme sulle sorgenti |
| `ltc/solver.py` | Aggiornamento semi-implicito e avanzamento per sottopassi |
| `ltc/validation.py` | Validazione di durata, forme, dispositivi e tipi numerici |
| `ltc/cell.py` | Cella LTC e stato iniziale coerente con dispositivo e tipo numerico |
| `ltc/sequence.py` | Elaborazione delle sequenze e raccolta degli stati |
| `data/temporal_order.py` | Generazione riproducibile di coppie di impulsi |
| `models/sequence_classifier.py` | Classificatore basato sullo stato finale |
| `visualization/synapse_curve.py` | Campionamento della curva sinaptica |
| `visualization/synapse_viewer.py` | Grafico interattivo con slider |
| `visualization/state_viewer.py` | Risposta dello stato a un impulso e confronto con ingresso nullo |
| `check_*.py` | Script di controllo |
| `.gitignore` | Esclusione di ambienti virtuali, cache e file locali |

## Limiti attuali

- `advance_state` valida durata, sottopassi e compatibilità dei tensori e dei parametri. `semi_implicit_step` valida la durata; chi lo chiama direttamente deve fornire tensori compatibili.
- Non sono ancora disponibili un ciclo di ottimizzazione, risultati di accuratezza o una valutazione su dati separati.
- `run_sequence` usa una durata scalare comune ai passi e raccoglie sempre tutta la storia, anche quando il classificatore utilizza solo lo stato finale.
- Le dipendenze non sono ancora fissate integralmente per riprodurre l'ambiente.
