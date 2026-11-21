import torch

from .event_windows import event_windows

# Ordine delle uscite del modello per il market making (Avellaneda-Stoikov).
OUTPUTS = ("drift", "volatility", "toxicity", "k")


def book_windows(events: dict[str, torch.Tensor], *, lookback: int,
                 with_agreement: bool = False) -> dict[str, torch.Tensor]:
    """Finestre di eventi del book con 4 obiettivi veri all'evento successivo.

    inputs  [n, lookback, 4 o 5]: rendimento, Δt, lato, portata. Il flag "informed" NON è un
            ingresso: conosce il movimento futuro.
    targets [n, 4]: drift, volatilità, tossicità e k veri all'evento successivo (ordine OUTPUTS).
    naive   [n, 4]: stime dalla sola finestra. Drift e volatilità come event_windows;
            k = 1 / portata media (massima verosimiglianza dell'esponenziale);
            tossicità = 2·(quota di lati concordi con il rendimento successivo) − 1, limitata a
            [0, 1], calcolata solo sulle coppie interne alla finestra (eventi già osservati).
    with_agreement: aggiunge un 5º ingresso, per l'evento i: lato_{i−1} · segno(r_i), cioè se
            l'ordine precedente aveva anticipato il movimento appena osservato (0 al primo evento).
            Usa solo dati già noti all'evento i.
    """
    base = event_windows(events, lookback=lookback)
    count = len(base["targets"])
    side = events["side"].double().unfold(0, lookback, 1)[:count]
    reach = events["reach"].double().unfold(0, lookback, 1)[:count]
    returns = events["returns"].double().unfold(0, lookback, 1)[:count]

    next_sign = torch.where(returns[:, 1:] >= 0, 1.0, -1.0)
    agreement = (side[:, :-1] == next_sign).double().mean(dim=1)
    naive_toxicity = (2 * agreement - 1).clamp(0.0, 1.0)
    naive_k = 1.0 / reach.mean(dim=1).clamp_min(1e-12)

    extra = [side, reach]
    if with_agreement:
        all_side = events["side"].double()
        all_sign = torch.where(events["returns"] >= 0, 1.0, -1.0).double()
        previous_side_agrees = torch.cat([torch.zeros(1, dtype=torch.float64),
                                          all_side[:-1] * all_sign[1:]])
        extra.append(previous_side_agrees.unfold(0, lookback, 1)[:count])

    return {
        "inputs": torch.cat([base["inputs"], torch.stack(extra, dim=2)], dim=2),
        "targets": torch.cat([base["targets"], torch.stack(
            [events["true_toxicity"][lookback:], events["true_k"][lookback:]], dim=1).double()], dim=1),
        "naive": torch.cat([base["naive"], torch.stack([naive_toxicity, naive_k], dim=1)], dim=1),
    }
