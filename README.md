# LTC RNN · Python

Repository del progetto **ltc-rnn-py**, sviluppato in Python.

Il progetto è nella fase iniziale di preparazione dell'ambiente. Gli obiettivi funzionali e l'architettura devono ancora essere definiti; al momento non sono presenti modelli, script di addestramento, dataset o risultati sperimentali.

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

## Contenuto del repository

- `README.md`: presentazione del progetto e istruzioni per l'ambiente.
- `.gitignore`: esclusioni per ambienti virtuali, cache, artefatti Python e file locali.
- `check_environment.py`: controllo dell'ambiente PyTorch e dell'accesso alla GPU.

L'implementazione dei modelli sarà aggiunta dopo la definizione delle specifiche.
