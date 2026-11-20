# Appunti sulle serie temporali finanziarie

Appunti dell'autore (2026-11-19), con lo stato attuale nel progetto.

## Stazionarietà
- Una serie è **debolmente stazionaria** se media e varianza non cambiano nel
  tempo e l'autocovarianza dipende solo dal ritardo, non dal momento; **fortemente stazionaria** se non cambia nemmeno la distribuzione
  congiunta delle osservazioni (tutte le statistiche di ordine superiore).
- Senza stazionarietà il modello deve prevedere in zone di valori che non ha mai
  visto in training: errori sistematici. Normalizzare con statistiche fisse non
  basta a rimediare.
- Il prezzo dell'S&P 500 **non è stazionario**: la media cresce nel tempo.
- **Nel progetto:** ingressi e obiettivi sono **rendimenti logaritmici**
  (differenza dei log-prezzi), non prezzi; standardizzati con media e deviazione
  del solo training. La media dei rendimenti è abbastanza stabile, la
  **varianza no** (anni '30, 1987, 2008, 2020 molto più agitati): i rendimenti
  sono solo in parte stazionari. Idea da provare: dividere ogni rendimento per la
  volatilità recente stimata sul passato (rendimenti "normalizzati per volatilità").

## Cosa prevedere e come misurarlo
- Prevedere il **prezzo** è un errore: conta il rendimento. Una previsione
  "prezzo di domani = prezzo di oggi" ha MSE piccolissimo ma non dice niente
  (naive forecast). **Nel progetto:** il riferimento "rendimento zero" è proprio
  questo naive forecast ed è sempre riportato.
- L'MSE può premiare una previsione nella **direzione sbagliata**: oggi 1000,
  domani 999, previsione 1000,2 → errore piccolo ma direzione opposta.
  **Nel progetto:** è riportata anche la frazione di segni indovinati.
- Spesso basta prevedere la **direzione** (su/giù). Attenzione al riferimento:
  l'S&P sale circa il 54 % dei giorni, quindi "sempre su" indovina già il 54 %.
  Una rete deve battere il 54 %, non il 50 %.
- Un solo giorno di storia non basta per prevedere il giorno dopo. **Nel
  progetto:** finestre di 30 giorni (`--lookback`).

## Modelli e addestramento
- Backpropagation (nel progetto: attraverso il tempo, BPTT). Forward-forward
  (Hinton) come alternativa senza backpropagation: da tenere presente, come il
  genetico del ramo droni.
- Problema delle dipendenze lunghe nelle RNN: i gradienti si attenuano su
  sequenze lunghe. La LTC con solver semi-implicito è più stabile; con lookback
  30 il problema è limitato, da verificare se si allunga la finestra.
- Softmax con temperatura: per un classificatore su/giù, la temperatura stimata
  sulla validazione rende le probabilità più affidabili (calibrazione).

## Fonti citate dall'autore
- Lazy Programmer, "common mistakes in stock prices", ep. 20 (YouTube).
- Dataquest, "Predict stock market with machine learning and Python" (YouTube).
