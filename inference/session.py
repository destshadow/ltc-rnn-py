from dataclasses import dataclass

import torch

from inference.stream import LTCStream


@dataclass(frozen=True)
class StreamSnapshot:
    index: int
    time: float
    sample: torch.Tensor
    state: torch.Tensor
    output: torch.Tensor

    def copy(self):
        return StreamSnapshot(
            index=self.index,
            time=self.time,
            sample=self.sample.clone(),
            state=self.state.clone(),
            output=self.output.clone(),
        )


class SequenceSession:
    """Riproduce una sequenza attraverso uno stream dedicato."""

    def __init__(self, stream: LTCStream, sequence: torch.Tensor):
        if sequence.ndim != 2:
            raise ValueError("sequence deve avere forma [istanti, ingressi].")

        if sequence.shape[0] == 0:
            raise ValueError("La sequenza non può essere vuota.")

        expected = stream.model.cell.config.input_size
        if sequence.shape[1] != expected:
            raise ValueError("Numero di ingressi non compatibile.")

        if not torch.is_floating_point(sequence):
            raise ValueError("La sequenza deve contenere numeri floating point.")

        if not torch.isfinite(sequence).all().item():
            raise ValueError("La sequenza contiene valori non finiti.")

        self._stream = stream
        self._sequence = sequence.detach().cpu().clone()
        self.reset()

    @property
    def total(self) -> int:
        return self._sequence.shape[0]

    @property
    def processed(self) -> int:
        return self._next_index

    @property
    def finished(self) -> bool:
        return self.processed >= self.total

    def reset(self) -> None:
        self._stream.reset()
        self._next_index = 0
        self._current = None

    def current(self) -> StreamSnapshot | None:
        if self._current is None:
            return None

        return self._current.copy()

    @torch.no_grad()
    def advance(self) -> StreamSnapshot:
        if self.finished:
            raise StopIteration("La sequenza è terminata.")

        index = self._next_index
        sample = self._sequence[index]
        output = self._stream.push(sample)
        state = self._stream.state_snapshot()

        if state is None:
            raise RuntimeError("Stato assente dopo l'elaborazione.")

        self._current = StreamSnapshot(
            index=index,
            time=(index + 1) * self._stream.dt,
            sample=sample.clone(),
            state=state[0].detach().cpu().clone(),
            output=output[0].detach().cpu().clone(),
        )
        self._next_index += 1

        return self.current()