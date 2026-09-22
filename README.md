# LTC RNN · Python

Progetto sperimentale in Python e PyTorch per costruire una rete ricorrente con dinamica a capacità e conduttanze sinaptiche, ispirata alle Liquid Time-Constant networks.

Il progetto comprende una cella LTC, un solver semi-implicito, un classificatore di sequenze e inferenza streaming con stato persistente. Sono disponibili dati sintetici sull'ordine temporale, addestramento, selezione del checkpoint su validazione, test separato e analisi della memoria e della soglia decisionale. Checkpoint e risultati restano locali nella cartella `outputs/`, esclusa da Git.

## Prima tappa completata — 2026-09-21

La prima tappa comprende LTC modulare, addestramento, valutazione separata, esportazione per inferenza, streaming con memoria persistente e visualizzazione interattiva.

**Limite attuale: riconoscimento di due ordini di eventi sintetici (A-B-C e B-A-C), con decisione valutata a fine sequenza.** I margini intermedi visualizzati sono provvisori; questi risultati non dimostrano riconoscimento continuo di fenomeni reali.

### Checkpoint di riferimento

| Voce | Riferimento |
| --- | --- |
| Checkpoint | `outputs/event_order_noise_20260921T150040_212480Z/best.pt` |
| Bundle della demo | `outputs/event_order_inference_1000ep.pt` |
| Epoca selezionata | 1000 |
| Configurazione | 3 ingressi, 8 neuroni nascosti, 6 sottopassi, `dt = 0.1` |
| Rumore di training / validazione | `0.0` / `0.01` |
| Soglia esportata | `-0.16052579879760742`, scelta sul training pulito su CPU |
| Versione PyTorch registrata | `2.13.0+cu130` |

SHA-256 del checkpoint, coincidente con quello registrato nel bundle e nel protocollo del test:

```text
27ad3b7c1beee3720bfd2290d3ff93a904dd8c4093de52fba3458585fce45dfb
```

Checkpoint, bundle e report restano locali in `outputs/`, esclusa da Git: il clone del repository non li contiene. Conservare questi artefatti per riprodurre la demo con gli stessi pesi; un nuovo training non garantisce un file identico.

### Risultati principali registrati

Risultati letti dal checkpoint e da `outputs/event_order_test_20260921T152429_193858Z/{protocol,results}.json`, senza rieseguire training o valutazione per questo aggiornamento. Il test usa 1000 coppie (2000 sequenze), seme 271828, CPU e rumore gaussiano additivo con semi 1101, 2202 e 3303. La soglia è fissata sul training, prima del test.

| Valutazione | Risultato |
| --- | --- |
| Validazione pulita, seme 2026 | 256/256 (100%), perdita `3.95424e-6` |
| Validazione con rumore `0.01`, seme 808 | 256/256 (100%), perdita `3.72378e-6` |
| Test pulito, soglia zero e soglia esportata | 2000/2000 (100%) |
| Test con rumore `0.001`, `0.005`, `0.01`, `0.02` | 2000/2000 per ciascun seme, con entrambe le soglie |
| Test con rumore `0.05`, soglia esportata | 2000/2000 per ciascun seme |
| Test con rumore `0.05`, soglia zero | 1999/2000 (99.95%) con seme 1101; 2000/2000 con gli altri due semi |

Sono risultati di un checkpoint su un insieme sintetico e sui semi indicati, non una garanzia per altri dati o livelli di rumore.

### Comandi della demo

Dalla radice del progetto, con il bundle locale disponibile:

```bash
source .venv/bin/activate
python visualize_stream.py --class-id 0 --seed 2026 --noise 0
python visualize_stream.py --class-id 1 --seed 2026 --noise 0
python visualize_stream.py --class-id 0 --seed 2026 --noise 0.05 --noise-seed 1101
```

Il percorso predefinito è `outputs/event_order_inference_1000ep.pt`; `--bundle <percorso>` lo sostituisce. `--seed` controlla la sequenza, `--noise` la deviazione standard del rumore e `--noise-seed` la sua realizzazione. Servono Matplotlib e un backend grafico interattivo.

**Pausa** ferma la riproduzione; **Un passo** mette in pausa e avanza di un campione; **Riprendi** continua dal successivo. **Ricomincia** svuota i grafici, azzera anche la memoria LTC tramite `session.reset()` e `stream.reset()` e resta in pausa. La precedente versione dei callback basata su `FuncAnimation` era stata verificata con una LTC su CPU e timer controllato su backend Agg. Tale verifica non copre il nuovo playback basato su `SequenceSession` né i clic sul backend GUI.

Per esportare nuovamente il riferimento, solo se il file di destinazione non esiste:

```bash
python prepare_inference.py --checkpoint outputs/event_order_noise_20260921T150040_212480Z/best.pt --output outputs/event_order_inference_1000ep.pt
```

### Organizzazione e prossime tappe

`SequenceSession` separa ora l'avanzamento della rete dalla grafica: `advance()` elabora un campione, `current()` restituisce una copia del risultato senza avanzare e `reset()` azzera sessione e memoria LTC. Il timer di `PlaybackControls` gestisce la riproduzione; grafici, neuroni e connessioni ricevono lo stesso snapshot.

Le prossime attività sono uniformare `--bundle` negli strumenti, registrare le dipendenze e completare l'osservazione della rete. Il percorso verso il controllo prevede poi simulatore verticale, controller classico di riferimento e nuovo controller LTC a uscita continua, valutato tramite algoritmo genetico. I comandi LTC andranno al simulatore; il genetico riceverà il punteggio della prova. Il classificatore A-B-C resta un esempio separato.

CSV e confronto file–generatore sono opzionali, da introdurre per registrare o riprodurre dati esterni; per A-B-C andrà verificata anche la corrispondenza del `dt` con il modello. Il simulatore potrà produrre direttamente le osservazioni. Per fenomeni reali serviranno dati, etichette e nuovo addestramento appropriati.

La struttura futura prevista distingue componenti autonomi `LTC/`, `GA/`, `WEB/` e `PROXY/`, quando necessari. Questo repository contiene attualmente il componente LTC e le sue visualizzazioni; le cartelle attuali non sono state trasferite né sono stati aggiunti gli altri componenti.

## Riattivare l'ambiente virtuale

A ogni nuovo terminale Bash/WSL, dalla cartella del progetto:

```bash
cd "/mnt/c/Users/amara/Desktop/LTC RNN py"
source .venv/bin/activate
```

Se sei già nella cartella, basta `source .venv/bin/activate`. Non serve ricreare l'ambiente né reinstallare le dipendenze. Il prompt normalmente mostra `(.venv)`; `which python` deve indicare l'interprete dentro `.venv/bin/`. Per uscire: `deactivate`. Questi comandi si riferiscono all'ambiente Linux/WSL del progetto.

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
python check_event_order.py
python check_sequence_modes.py
python check_stream.py
```

`check_config.py`, `check_solver_validation.py`, `check_solver_dynamics.py`, `check_temporal_order.py`, `check_event_order.py` e `check_sequence_modes.py` funzionano su CPU. `check_cell.py` aggiunge un confronto CPU/GPU quando CUDA è disponibile; `check_classifier.py` e `check_stream.py` scelgono CUDA se disponibile, altrimenti CPU. `check_stream.py` richiede `outputs/event_order_best.pt`. Gli altri script elencati richiedono CUDA nella loro versione attuale. Questa lista documenta i controlli disponibili, non attesta che siano stati tutti eseguiti e superati.

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
| `check_event_order.py` | Struttura e riproducibilità delle coppie A-B-C/B-A-C |
| `check_sequence_modes.py` | Equivalenza di stati finali e gradienti con e senza raccolta della storia |
| `check_stream.py` | Equivalenza tra sequenza intera, blocchi e singoli campioni; reset |

Il test del solver verifica che cento sottopassi siano più accurati di un singolo passo nel caso di decadimento. Non costituisce una verifica completa della dinamica ricorrente o dell'addestramento.

## Sequenze e classificazione

`LTCCell` raggruppa i parametri dei neuroni e le sinapsi sensoriali e ricorrenti. Lo stato iniziale segue dispositivo e tipo numerico della cella. `run_sequence` elabora ingressi `[batch, istanti, ingressi]` e restituisce `(history, final_state)`, mantenendo il percorso dei gradienti. Accetta uno stato iniziale esplicito e `collect_history`: con `True` raccoglie gli stati; con `False` restituisce `None` come storia. Il classificatore usa `False`. Questo evita la raccolta e lo stacking espliciti, ma non elimina le informazioni necessarie ad autograd durante l'addestramento.

`SequenceClassifier` applica `nn.Linear` allo stato finale e restituisce logits utilizzabili con `CrossEntropyLoss`. `check_classifier.py` calcola una perdita e chiama `backward()`, ma non aggiorna i pesi: le previsioni stampate sono quelle del modello non addestrato.

`make_temporal_order(pairs, seed=...)` genera `2 * pairs` sequenze di 30 passi, ciascuna con due impulsi opposti. La classe 0 presenta prima l'impulso positivo, la classe 1 quello negativo. Ogni coppia condivide posizioni e ampiezza; gli ultimi campioni sono sempre zero. Il seme rende riproducibile la generazione.

Il compito richiede conservare informazione dagli impulsi precedenti, ma può essere risolto ricordando il segno dell'ultimo impulso non nullo: non misura da solo capacità temporali generali.

Il secondo dataset, `make_event_order`, genera coppie A-B-C/B-A-C su tre canali e 48 passi. Posizioni e ampiezza sono condivise nella coppia, l'evento finale C è identico e l'attesa successiva varia da 0 a 30 passi. La classe 0 corrisponde ad A-B-C, la classe 1 a B-A-C.

## Addestramento e analisi

Tutti i comandi vanno eseguiti dalla radice, con `.venv` attivo. Gli script di training scelgono CUDA se disponibile, altrimenti CPU; il test separato e l’esportazione per inferenza usano la CPU. I checkpoint non sono inclusi nel repository: generarli prima delle analisi che li richiedono.

### Esperimento con due impulsi

```bash
python train_tiny.py
python evaluate_temporal_order.py
python inspect_temporal_errors.py
python -m visualization.memory_grid_viewer
```

Il training salva `outputs/tiny_checkpoint.pt`. Gli altri comandi caricano quel file per valutare il modello, analizzare gli errori e visualizzare l'effetto di ampiezza e attesa sulla memoria. Il viewer richiede un backend grafico interattivo.

### Esperimento A-B-C/B-A-C

```bash
python train_event_order.py
python inspect_event_memory.py
python inspect_decision_threshold.py
python test_event_order.py --checkpoint outputs/event_order_best.pt
```

Il training usa mini-batch, Adam, clipping dei gradienti e insiemi generati con semi distinti: 123 per training, 2026 per validazione. Valuta ogni dieci epoche, includendo l'epoca zero, e salva il checkpoint con la minore perdita tra le epoche valutate in `outputs/event_order_best.pt`. Una nuova esecuzione può sovrascrivere questo file.

- `inspect_event_memory.py` analizza gli stati delle coppie prima e dopo C e alla fine, raggruppando le distanze per attesa; riporta anche margini e classificazioni sui dati di validazione.
- `inspect_decision_threshold.py` sceglie la soglia esclusivamente sul training e confronta training e validazione, senza modificare il checkpoint.
- `test_event_order.py --checkpoint <percorso>` usa la CPU e sceglie la soglia sul training pulito prima di generare il test: 1000 coppie con seme 271828, distinto dai semi di training e validazione. Confronta soglia zero e soglia scelta su dati puliti e con rumore a livelli `0.001`, `0.005`, `0.01`, `0.02`, `0.05`, con semi 1101/2202/3303. Riporta risultati per attesa e salva il protocollo prima della valutazione e i risultati progressivamente in `outputs/event_order_test_<timestamp>/`. Il protocollo comprende hash SHA-256, epoca del checkpoint, configurazione, dispositivo e versione PyTorch.

La decisione binaria usa il margine `logit_0 - logit_1`: sotto la soglia assegna classe 1, altrimenti classe 0. La soglia scelta dagli script non viene applicata automaticamente al modello o allo stream. I risultati salvati del checkpoint di riferimento sono riportati nella sezione sulla prima tappa.

### Esperimento con rumore configurabile

```bash
python train_event_order_noise.py
python compare_noise_models.py --noisy-checkpoint outputs/event_order_noise_<timestamp>/best.pt
```

`train_event_order_noise.py` usa attualmente `training_noise_std = 0.0` e `validation_noise_std = 0.01`: il training è pulito, mentre la validazione comprende sia dati puliti sia una copia con rumore gaussiano fisso (seme 808). Restano 1000 epoche, valutazione ogni dieci epoche inclusa l'epoca zero, Adam con learning rate `0.01`, batch di 64, `dt = 0.1`, 8 neuroni nascosti e 6 sottopassi. I dataset contengono 128 coppie ciascuno, con semi 123/2026; il seme del modello è 42 e quello del rumore di training è 707.

Il criterio di selezione resta la media delle perdite di validazione pulita e rumorosa. Ogni esecuzione crea `outputs/event_order_noise_<timestamp>/` con `metrics.csv` e `best.pt`; il checkpoint registra entrambi i valori effettivi in `training_noise_std` e `validation_noise_std`, oltre ai semi e alle metriche. La modifica vale per le nuove esecuzioni e non altera i checkpoint degli esperimenti conclusi.

`compare_noise_models.py` confronta il checkpoint indicato con `outputs/event_order_best.pt` sulla validazione (seme 2026), scegliendo separatamente le soglie sul training pulito. Usa livelli di rumore `0.0`, `0.001`, `0.005`, `0.01`, `0.02`, `0.05` e semi 101/202/303, condivisi tra i modelli; salva risultati e hash in `outputs/noise_comparison_<timestamp>/results.json`. Sostituire `<timestamp>` con la cartella dell'esperimento. Il parametro `--noisy-checkpoint` accetta anche il nuovo esperimento con training pulito; questo confronto usa la validazione, non il test finale.

## Inferenza streaming

`LTCStream(model, dt=...)` conserva lo stato di un singolo flusso e imposta il modello in modalità valutazione:

| Metodo | Ingresso / comportamento |
| --- | --- |
| `push(sample)` | Un campione `[input_size]` |
| `push_block(samples)` | Un blocco non vuoto `[istanti, input_size]` |
| `reset()` | Elimina lo stato per iniziare un flusso indipendente |
| `state_snapshot()` | Copia indipendente dello stato, senza gradienti; `None` prima del primo campione o dopo il reset |

Entrambi i metodi di inserimento restituiscono logits `[1, num_classes]` dopo l'ultimo campione ricevuto, senza registrare gradienti. Gli ingressi vengono convertiti al dispositivo e al tipo numerico del modello. Lo stream non applica soglie né restituisce probabilità: mantiene la continuità dello stato tra chiamate, fino al reset.

### Esportazione, decisione e misure

```bash
python prepare_inference.py --checkpoint outputs/event_order_best.pt --output outputs/event_order_inference.pt
python benchmark_stream.py
python inspect_noise_robustness.py
```

`prepare_inference.py` richiede `--checkpoint` e `--output`, controlla che il modello abbia due classi e sceglie la soglia sul training in modalità valutazione su CPU. Esporta pesi, configurazione, soglia, percorso e hash del checkpoint, epoca sorgente, dispositivo della soglia e versione PyTorch. Crea le cartelle necessarie e salva con apertura esclusiva (`xb`), senza sovrascrivere file esistenti. I due comandi successivi richiedono `outputs/event_order_inference.pt`.

`inference/loading.py` carica il bundle; `inference/decision.py` applica la soglia ai logits binari. `check_stream_decision.py` controlla equivalenza delle decisioni tra sequenza intera e streaming e correttezza sulle quattro sequenze generate. `benchmark_stream.py` misura la latenza per campione inclusa la decisione, su CPU e su CUDA se disponibile, riportando mediana, 95° percentile e massimo.

`inspect_noise_robustness.py` valuta su CPU il bundle con la soglia salvata, sui dati di validazione e sui livelli di rumore e semi descritti sopra. Salva il report in `outputs/noise_validation_<timestamp>/results.json`. Questi comandi documentano gli strumenti disponibili; non attestano risultati verificati.

`check_stream_decision.py` usa invece il percorso fisso `outputs/event_order_inference_1000ep.pt`. Per prepararlo dal checkpoint desiderato:

```bash
python prepare_inference.py --checkpoint outputs/event_order_noise_<timestamp>/best.pt --output outputs/event_order_inference_1000ep.pt
python check_stream_decision.py
```

Sostituire `<timestamp>` con la cartella reale. Il suffisso `1000ep` è un nome di file: il checkpoint migliore può provenire da un'epoca precedente alla millesima. Il controllo richiede anche che tutte e quattro le sequenze siano classificate correttamente; un errore di accuratezza non dimostra da solo un errore nello streaming.

## Visualizzazioni

### Neuroni e connessioni ricorrenti

```bash
python visualize_stream.py --connections strength
python visualize_stream.py --connections conductance --class-id 1 --seed 2026
```

`--connections` accetta `strength` (predefinito) e `conductance`. Il colore dei neuroni indica lo stato interno, su scala fissa; facendo clic su un neurone si selezionano le connessioni ricorrenti entranti. L'autoconnessione è riportata numericamente, senza freccia.

- `strength`: spessore proporzionale all'intensità appresa, costante durante la riproduzione.
- `conductance`: spessore aggiornato dalle conduttanze calcolate sullo stato dello snapshot visualizzato, senza avanzare la LTC. Non rappresenta la storia delle conduttanze nei sottopassi del solver.

Le modalità condividono la scala basata sulla massima `strength`. Le frecce sono neutre: una conduttanza positiva non implica un aumento dello stato del neurone; l'effetto dipende anche dal potenziale di inversione e dallo stato della destinazione. Questa vista mostra le connessioni ricorrenti, non ancora quelle sensoriali o il dettaglio completo di una singola sinapsi.

**Ricomincia** azzera grafici, colori dei neuroni e valori delle connessioni riferiti allo stato nullo, mantenendo il neurone selezionato. In modalità conduttanza, stato nullo non significa necessariamente conduttanza nulla.

`check_session.py --bundle <percorso>` controlla lettura senza avanzamento, indipendenza degli snapshot, equivalenza con la sequenza intera, fine sequenza e reset. Lo script è disponibile ma non è stato eseguito in questo aggiornamento.


```bash
python visualize_stream.py --bundle outputs/event_order_inference_1000ep.pt --class-id 0
python visualize_stream.py --bundle outputs/event_order_inference_1000ep.pt --class-id 1
```

Il viewer usa la CPU e anima una sequenza sintetica A-B-C o B-A-C (`--seed`, predefinito 2026), con rumore opzionale (`--noise`, predefinito 0.0; `--noise-seed`, predefinito 1101), mostrando ingressi, stato dei neuroni e margine rispetto alla soglia salvata. Le decisioni intermedie sono indicate come provvisorie: il classificatore è addestrato sullo stato finale. Richiede il bundle esportato e un backend grafico interattivo di Matplotlib. `--bundle` permette di usare un percorso diverso.


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
| `data/event_order.py` | Coppie A-B-C/B-A-C con tempi variabili |
| `data/noise.py` | Rumore gaussiano con generatore CPU esplicito |
| `data/memory_grid.py` | Griglia di ampiezze e attese per l'analisi della memoria |
| `models/sequence_classifier.py` | Classificatore basato sullo stato finale |
| `training/step.py`, `training/epoch.py` | Aggiornamento dei pesi e training per mini-batch |
| `training/evaluation.py` | Perdita, accuratezza e matrice di confusione binaria |
| `training/decision.py` | Raccolta dei margini e selezione della soglia |
| `inference/session.py` | Avanzamento della sequenza e snapshot indipendenti dalla grafica |
| `inference/connection_inspection.py` | Calcolo delle conduttanze ricorrenti sullo stato osservato |
| `inference/stream.py` | Inferenza con stato persistente tra campioni o blocchi |
| `inference/loading.py`, `inference/decision.py` | Caricamento del bundle e decisione con soglia |
| `prepare_inference.py`, `benchmark_stream.py` | Esportazione del bundle e misura della latenza |
| `compare_noise_models.py` | Confronto dei checkpoint su validazione con rumore condiviso |
| `visualization/synapse_curve.py` | Campionamento della curva sinaptica |
| `visualization/synapse_viewer.py` | Grafico interattivo con slider |
| `visualization/state_viewer.py` | Risposta dello stato a un impulso e confronto con ingresso nullo |
| `visualization/memory_grid_viewer.py` | Visualizzazione della memoria del modello addestrato sui due impulsi |
| `train_*.py`, `evaluate_temporal_order.py`, `inspect_*.py`, `test_event_order.py` | Esperimenti, valutazione e analisi dei checkpoint |
| `visualize_stream.py` | Caricamento del bundle, preparazione dei dati e animazione streaming |
| `visualization/stream_plot.py` | Creazione dei pannelli e spazio per i controlli |
| `visualization/playback.py` | Timer, pausa, avanzamento della sessione e riavvio |
| `visualization/neuron_view.py` | Stato dei neuroni tramite colori e valori |
| `visualization/neuron_connections.py` | Selezione del neurone e connessioni ricorrenti per strength o conduttanza |
| `check_*.py` | Script di controllo |
| `.gitignore` | Esclusione di ambienti virtuali, cache e file locali |

## Limiti attuali

- `advance_state` valida durata, sottopassi e compatibilità dei tensori e dei parametri. `semi_implicit_step` valida la durata; chi lo chiama direttamente deve fornire tensori compatibili.
- Gli esperimenti usano dati sintetici; non dimostrano da soli generalizzazione a dati reali o a tempi fuori dalla distribuzione di training.
- Il test separato usa un seme fisso: riutilizzarlo per scegliere modifiche al modello ne comprometterebbe il ruolo di valutazione finale.
- `run_sequence` usa una durata scalare comune ai passi; `LTCStream` gestisce un solo flusso per istanza, con `dt` fissato alla creazione.
- Valutazione e scelta della soglia attuali sono pensate per due classi.
- Le dipendenze non sono ancora fissate integralmente per riprodurre l'ambiente.
