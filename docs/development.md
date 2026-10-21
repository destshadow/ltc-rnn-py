# Sviluppo e verifiche

Il progetto dichiara Python >= 3.10 e le dipendenze in `pyproject.toml`.
I limiti inferiori sono requisiti dichiarati, non una certificazione di tutte
le combinazioni di versioni. La CI controlla Python 3.10 e 3.12 su CPU.

## Installazione

Dalla radice, nell'ambiente virtuale scelto:

```bash
python -m pip install -e '.[dev]'
```

Questo installa il progetto e pytest. Per un nuovo ambiente solo CPU,
installare prima PyTorch dall'indice CPU:

```bash
python -m pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

Un PyTorch CUDA già installato che soddisfa il requisito può essere conservato.
Il progetto non impone una versione CUDA a tutti gli ambienti.

## Suite automatica

```bash
python -m pytest
```

La raccolta è limitata a `tests/`: `test_event_order.py` nella radice rimane
un esperimento separato che richiede un checkpoint addestrato.
La suite raccoglie 11 controlli esistenti portabili e sei controlli CUDA,
saltati quando CUDA non è disponibile. Alcuni controlli portabili aggiungono
automaticamente un confronto GPU se disponibile. Per forzare solo CPU:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest
```

I nuovi test controllano il round-trip del controller in float32 e float64,
la protezione dalla sovrascrittura, il rifiuto di dtype non supportati o misti,
metadati incompleti/incompatibili e pesi non finiti o con forme errate.
Streaming e sessione sono confrontati con l'esecuzione completa usando pesi
casuali: non occorrono file in `outputs/`, training o un backend grafico.

I controlli standalone restano disponibili. La suite non verifica l'accuratezza
dei checkpoint addestrati né le interazioni GUI. Evitare `python -O` per
eseguire i controlli standalone basati su assert.

La workflow `.github/workflows/tests.yml` esegue la suite su push, pull request
e avvio manuale. Non è stata eseguita durante questa modifica; l'esecuzione
locale dei test resta affidata all'utente secondo la preferenza registrata.

## Ambiente locale osservato

Versioni lette il 2026-10-21 dai file METADATA del venv esistente, senza importare
le librerie o verificare la disponibilità delle wheel sugli indici:

| Libreria | Versione locale |
| --- | --- |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.2 |

Questa tabella è un inventario parziale, non un lockfile. Per conservare
l'ambiente funzionante l'utente può registrare tutte le dipendenze con
`python -m pip freeze`; la scelta dell'indice PyTorch va conservata insieme
all'inventario. La CI installa versioni compatibili risolte al momento.

## Bundle del controller

Il formato resta versione 1. Salvataggio e caricamento accettano soltanto
float32 e float64 uniformi. Il loader valida i campi obbligatori, configurazione,
dimensione dell'uscita, dtype e pesi prima di restituire il modello.
Metadati extra sono ammessi per compatibilità futura.
La memoria episodica non viene salvata; normalizzazione e conversione della
spinta restano da definire con il simulatore.

## Coordinamento locale

`.coordination/` è esclusa da Git. Ogni agente può scrivere un proprio file con
ambito, file riservati e stato; prima di modificare un file condiviso controllare
gli altri registri e il diff corrente. Il registro è un accordo fra agenti,
non un lock automatico. I push sono eseguiti dall'utente.
