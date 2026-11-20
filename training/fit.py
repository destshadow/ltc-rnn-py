import copy

from .epoch import train_epoch


def fit_and_select(
    model, optimizer, loss_function, training, validation, *,
    dt: float, epochs: int, batch_size: int, final_epoch: bool, metrics, save_best, predict,
):
    """Addestra per `epochs` epoche e restituisce (epoca scelta, pesi scelti).

    training, validation: coppie (ingressi, obiettivi); la validazione viene valutata
    a ogni epoca e scritta in `metrics`. Si sceglie l'epoca con la validazione migliore,
    oppure l'ultima con final_epoch. save_best(pesi, epoca, loss) è chiamata a ogni
    nuova scelta, così un'interruzione non perde il modello.
    """
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs + 1):
        train_loss = None
        if epoch > 0:
            train_loss = train_epoch(model, optimizer, loss_function, *training,
                                     dt=dt, batch_size=batch_size)

        validation_loss = loss_function(predict(model, validation[0], dt=dt), validation[1]).item()
        metrics.write(epoch, train_loss, validation_loss)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoca {epoch:4d} | val mse (standardizzato) {validation_loss:.4f}", flush=True)

        if validation_loss < best_loss or final_epoch:  # con final_epoch: l'ultima
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            save_best(best_state, epoch, validation_loss)

    return best_epoch, best_state
